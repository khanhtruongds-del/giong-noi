// Mandarin grapheme-to-phoneme for Kokoro-82M-v1.1-zh, matching misaki ZHG2P(version='1.1') (ZHFrontend):
//   cn2an an2cn -> punctuation mapping -> jieba.posseg -> ToneSandhi.pre_merge_for_modify
//   -> pypinyin (+ pypinyin_dict large_pinyin) initials/finals -> tone sandhi -> erhua -> zhuyin-like symbols.
// Sources ported: misaki (Apache-2.0, adapted from PaddleSpeech, Apache-2.0), jieba posseg (MIT),
// pypinyin + pypinyin-dict (MIT), cn2an (MIT); jieba segmentation via jieba-wasm (MIT).
import initJieba, { cut, cut_for_search } from './jieba_rs_wasm.js';
import { transformAn2Cn, mapPunctuation, tone3 } from './zh.js';

let CH, PH, PREFIX, SYL, TAGS, FREQ, HMM, NUMX, SYLB, ready = null;

const lines = (t) => t.split('\n').filter(Boolean);
export function loadZh11(base) {
  if (!ready) ready = (async () => {
    const get = (f) => fetch(base + f).then(r => { if (!r.ok) throw new Error(f); return f.endsWith('.json') ? r.json() : r.text(); });
    const [c, p, s, t, h, m, b] = await Promise.all(['py_chars.txt', 'py_phrases.txt', 'py_syl.json', 'jieba_tags.txt', 'pos_hmm.json', 'misc.json', 'syllables.json'].map(get));
    SYLB = b;
    CH = new Map(lines(c).map(l => l.split('\t')));
    PH = new Map(lines(p).map(l => { const i = l.indexOf('\t'); return [l.slice(0, i), l.slice(i + 1).split(' ')]; }));
    PREFIX = new Set();
    for (const k of PH.keys()) { const a = [...k]; for (let n = 1; n <= a.length; n++) PREFIX.add(a.slice(0, n).join('')); }
    SYL = s; TAGS = new Map(); FREQ = new Set();
    for (const l of lines(t)) { const a = l.split(' '); TAGS.set(a[0], a[1]); if (a.length < 3) FREQ.add(a[0]); }
    HMM = h; HMM.states = h.S.map(x => x.split(',')); HMM.trans = h.trans.map(o => new Map(Object.entries(o).map(([k, v]) => [+k, v])));
    NUMX = new Set([...m.numeric_extra]);
    await initJieba({ module_or_path: base + 'jieba_rs_wasm_bg.wasm' });
  })();
  return ready;
}

const chars = (s) => [...s];
const isNumeric = (c) => /\p{N}/u.test(c) || NUMX.has(c);
const isSpace = (s) => s.length > 0 && /^\s+$/u.test(s);

// ---- jieba.posseg.lcut ----------------------------------------------------------------------------
const MIN_FLOAT = -3.14e100;
const RE_HAN_DETAIL = /([一-鿕]+)/u, RE_SKIP_DETAIL = /([.0-9]+|[a-zA-Z0-9]+)/u;
const RE_HAN_INTERNAL = /([一-鿕a-zA-Z0-9+#&._]+)/u, RE_SKIP_INTERNAL = /(\r\n|\s)/u;
const RE_ENG = /^[a-zA-Z0-9]+/, RE_NUM = /^[.0-9]+/;
const whole = (re) => new RegExp('^' + re.source.slice(1, -1) + '$', 'u');
const HAN_DETAIL_FULL = /^[一-鿕]/u, HAN_INTERNAL_START = /^[一-鿕a-zA-Z0-9+#&._]/u, SKIP_INTERNAL_START = /^(\r\n|\s)/u;
void whole;

function viterbi(obs) {
  const { states, start, emit, trans } = HMM;
  const all = states.map((_, i) => i);
  const st = (c) => HMM.chars[c] || all;
  const em = (y, c) => { const v = emit[y][c]; return v === undefined ? MIN_FLOAT : v; };
  let V = new Map(), path = [];
  let mem = new Map();
  for (const y of st(obs[0])) { V.set(y, start[y] + em(y, obs[0])); mem.set(y, -1); }
  path.push(mem);
  for (let t = 1; t < obs.length; t++) {
    const prev = [...path[t - 1].keys()].filter(x => trans[x].size > 0);
    const expect = new Set(); for (const x of prev) for (const y of trans[x].keys()) expect.add(y);
    let obsStates = st(obs[t]).filter(y => expect.has(y));
    if (!obsStates.length) obsStates = expect.size ? [...expect] : all;
    const V2 = new Map(), m2 = new Map();
    for (const y of obsStates) {
      let best = -Infinity, bs = -1; const e = em(y, obs[t]);
      for (const y0 of prev) {
        const tp = trans[y0].get(y);
        const p = V.get(y0) + (tp === undefined ? -Infinity : tp) + e;
        if (bs < 0 || p > best || (p === best && y0 > bs)) { best = p; bs = y0; }
      }
      V2.set(y, best); m2.set(y, bs);
    }
    V = V2; path.push(m2);
  }
  let best = -Infinity, state = -1;
  for (const y of path[path.length - 1].keys()) { const p = V.get(y); if (state < 0 || p > best || (p === best && y > state)) { best = p; state = y; } }
  const route = new Array(obs.length);
  for (let i = obs.length - 1; i >= 0; i--) { route[i] = states[state]; state = path[i].get(state); }
  return route;
}
function* posCut(s) {
  const a = chars(s), r = viterbi(a);
  let begin = 0, next = 0;
  for (let i = 0; i < a.length; i++) {
    const p = r[i][0];
    if (p === 'B') begin = i;
    else if (p === 'E') { yield [a.slice(begin, i + 1).join(''), r[i][1]]; next = i + 1; }
    else if (p === 'S') { yield [a[i], r[i][1]]; next = i + 1; }
  }
  if (next < a.length) yield [a.slice(next).join(''), r[next][1]];
}
function* cutDetail(s) {
  for (const blk of s.split(RE_HAN_DETAIL)) {
    if (HAN_DETAIL_FULL.test(blk)) yield* posCut(blk);
    else for (const x of blk.split(RE_SKIP_DETAIL)) if (x) yield [x, RE_NUM.test(x) ? 'm' : RE_ENG.test(x) ? 'eng' : 'x'];
  }
}
function* flushBuf(buf) {
  const n = chars(buf).length;
  if (n === 1) yield [buf, TAGS.get(buf) || 'x'];
  else if (!FREQ.has(buf)) yield* cutDetail(buf);
  else for (const c of buf) yield [c, TAGS.get(c) || 'x'];
}
function* cutDAG(blk) {
  let buf = '';
  for (const w of cut(blk, false)) {
    // jieba-rs merges runs of ASCII letters/digits that Python routes one char at a time
    const singles = chars(w).length === 1 || (/^[a-zA-Z0-9]+$/.test(w) && !FREQ.has(w));
    if (singles) { buf += w; continue; }
    if (buf) { yield* flushBuf(buf); buf = ''; }
    yield [w, TAGS.get(w) || 'x'];
  }
  if (buf) yield* flushBuf(buf);
}
function psgCut(sentence) {
  const out = [];
  for (const blk of sentence.split(RE_HAN_INTERNAL)) {
    if (HAN_INTERNAL_START.test(blk)) { for (const p of cutDAG(blk)) out.push(p); continue; }
    for (const x of blk.split(RE_SKIP_INTERNAL)) {
      if (SKIP_INTERNAL_START.test(x)) { out.push([x, 'x']); continue; }
      for (const xx of x) out.push([xx, RE_NUM.test(xx) ? 'm' : RE_ENG.test(x) ? 'eng' : 'x']);
    }
  }
  return out;
}

// ---- pypinyin lazy_pinyin(style=INITIALS / FINALS_TONE3, neutral_tone_with_five=True, strict=True) --
const RE_HANS = /^[〇-礼㐀-䶿一-鿿豈-﫿\u{20000}-\u{2A6DF}\u{2A703}-\u{2B73F}\u{2B740}-\u{2B81D}\u{2B825}-\u{2BF6E}\u{2C029}-\u{2CE93}\u{2D016}\u{2D11B}-\u{2EBD9}\u{2F80A}-\u{2FA1F}\u{30000}-\u{3134A}\u{300F7}-\u{31288}\u{30EDD}\u{30EDE}\u{31350}-\u{32389}]+$/u;
function* mmseg(text) {   // pypinyin.seg.mmseg.Seg(no_non_phrases=True).cut
  let remain = chars(text);
  while (remain.length) {
    let lastValid = '', lastIdx = 0, broke = false;
    for (let i = 0; i < remain.length; i++) {
      const w = remain.slice(0, i + 1).join('');
      if (PREFIX.has(w)) { if (PH.has(w)) { lastValid = w; lastIdx = i + 1; } }
      else {
        if (lastValid) { yield lastValid; remain = remain.slice(lastIdx); } else { yield remain[0]; remain = remain.slice(1); }
        broke = true; break;
      }
    }
    if (!broke) {
      if (lastValid) { yield lastValid; remain = remain.slice(lastIdx); }
      else { const r = remain.join(''); if (PH.has(r)) yield r; else yield* remain; remain = []; }
    }
  }
}
function simpleSeg(s) {
  const out = []; let cur = '', flag = -1;
  for (const c of s) {
    const h = RE_HANS.test(c) ? 0 : 1;
    if (flag === -1 || h === flag) cur += c; else { out.push(cur); cur = c; }
    flag = h;
  }
  out.push(cur);
  return out;
}
// returns [[initial, finalTone3], ...] — non-pinyin runs come back unchanged in both slots
function lazyPinyin(word) {
  const out = [];
  for (const run of simpleSeg(word)) {
    if (!RE_HANS.test(run)) { if (run) out.push([run, run]); continue; }
    for (const piece of mmseg(run)) {
      const raws = PH.has(piece) ? PH.get(piece) : chars(piece).map(c => CH.get(c) ?? null);
      chars(piece).forEach((c, i) => { const r = raws[i]; out.push(r == null ? [c, c] : (SYL[r] || [r, r])); });
    }
  }
  return out;
}

// ---- misaki ToneSandhi --------------------------------------------------------------------------
const MUST_NEURAL = new Set('麻烦 麻利 鸳鸯 高粱 骨头 骆驼 马虎 首饰 馒头 馄饨 风筝 难为 队伍 阔气 闺女 门道 锄头 铺盖 铃铛 铁匠 钥匙 里脊 里头 部分 那么 道士 造化 迷糊 连累 这么 这个 运气 过去 软和 转悠 踏实 跳蚤 跟头 趔趄 财主 豆腐 讲究 记性 记号 认识 规矩 见识 裁缝 补丁 衣裳 衣服 衙门 街坊 行李 行当 蛤蟆 蘑菇 薄荷 葫芦 葡萄 萝卜 荸荠 苗条 苗头 苍蝇 芝麻 舒服 舒坦 舌头 自在 膏药 脾气 脑袋 脊梁 能耐 胳膊 胭脂 胡萝 胡琴 胡同 聪明 耽误 耽搁 耷拉 耳朵 老爷 老实 老婆 戏弄 将军 翻腾 罗嗦 罐头 编辑 结实 红火 累赘 糨糊 糊涂 精神 粮食 簸箕 篱笆 算计 算盘 答应 笤帚 笑语 笑话 窟窿 窝囊 窗户 稳当 稀罕 称呼 秧歌 秀气 秀才 福气 祖宗 砚台 码头 石榴 石头 石匠 知识 眼睛 眯缝 眨巴 眉毛 相声 盘算 白净 痢疾 痛快 疟疾 疙瘩 疏忽 畜生 生意 甘蔗 琵琶 琢磨 琉璃 玻璃 玫瑰 玄乎 狐狸 状元 特务 牲口 牙碜 牌楼 爽快 爱人 热闹 烧饼 烟筒 烂糊 点心 炊帚 灯笼 火候 漂亮 滑溜 溜达 温和 清楚 消息 浪头 活泼 比方 正经 欺负 模糊 槟榔 棺材 棒槌 棉花 核桃 栅栏 柴火 架势 枕头 枇杷 机灵 本事 木头 木匠 朋友 月饼 月亮 暖和 明白 时候 新鲜 故事 收拾 收成 提防 挖苦 挑剔 指甲 指头 拾掇 拳头 拨弄 招牌 招呼 抬举 护士 折腾 扫帚 打量 打算 打扮 打听 打发 扎实 扁担 戒指 懒得 意识 意思 悟性 怪物 思量 怎么 念头 念叨 别人 快活 忙活 志气 心思 得罪 张罗 弟兄 开通 应酬 庄稼 干事 帮手 帐篷 希罕 师父 师傅 巴结 巴掌 差事 工夫 岁数 屁股 尾巴 少爷 小气 小伙 将就 对头 对付 寡妇 家伙 客气 实在 官司 学问 字号 嫁妆 媳妇 媒人 婆家 娘家 委屈 姑娘 姐夫 妯娌 妥当 妖精 奴才 女婿 头发 太阳 大爷 大方 大意 大夫 多少 多么 外甥 壮实 地道 地方 在乎 困难 嘴巴 嘱咐 嘟囔 嘀咕 喜欢 喇嘛 喇叭 商量 唾沫 哑巴 哈欠 哆嗦 咳嗽 和尚 告诉 告示 含糊 吓唬 后头 名字 名堂 合同 吆喝 叫唤 口袋 厚道 厉害 千斤 包袱 包涵 匀称 勤快 动静 动弹 功夫 力气 前头 刺猬 刺激 别扭 利落 利索 利害 分析 出息 凑合 凉快 冷战 冤枉 冒失 养活 关系 先生 兄弟 便宜 使唤 佩服 作坊 体面 位置 似的 伙计 休息 什么 人家 亲戚 亲家 交情 云彩 事情 买卖 主意 丫头 丧气 两口 东西 东家 世故 不由 下水 下巴 上头 上司 丈夫 丈人 一辈 那个 菩萨 父亲 母亲 咕噜 邋遢 费用 冤家 甜头 介绍 荒唐 大人 泥鳅 幸福 熟悉 计划 扑腾 蜡烛 姥爷 照顾 喉咙 吉他 弄堂 蚂蚱 凤凰 拖沓 寒碜 糟蹋 倒腾 报复 逻辑 盘缠 喽啰 牢骚 咖喱 扫把 惦记'.split(' '));
const MUST_NOT_NEURAL = new Set('男子 女子 分子 原子 量子 莲子 石子 瓜子 电子 人人 虎虎 幺幺 干嘛 学子 哈哈 数数 袅袅 局地 以下 娃哈哈 花花草草 留得 耕地 想想 熙熙 攘攘 卵子 死死 冉冉 恳恳 佼佼 吵吵 打打 考考 整整 莘莘 落地 算子 家家户户 青青'.split(' '));
const SANDHI_PUNC = "、：，；。？！“”‘’':,;.?!";
const X_ENG = new Set(['x', 'eng']);
const setTone = (f, t) => f.slice(0, -1) + t;
const last = (s) => s.slice(-1);
const allThree = (fs) => fs.every(x => last(x) === '3');

function splitWord(word) {
  const list = cut_for_search(word, true).slice().sort((a, b) => chars(a).length - chars(b).length);
  const first = list[0], w = chars(word), n = chars(first).length;
  return word.indexOf(first) === 0 ? [first, w.slice(n).join('')] : [w.slice(0, w.length - n).join(''), first];
}
function neuralSandhi(word, pos, finals) {
  if (MUST_NOT_NEURAL.has(word)) return finals;
  const w = chars(word);
  for (let j = 1; j < w.length; j++) if (w[j] === w[j - 1] && 'nva'.includes(pos[0])) finals[j] = setTone(finals[j], '5');
  const geIdx = w.indexOf('个'), L = w.length, end = w[L - 1], end2 = w.slice(-2).join('');
  if (L >= 1 && '吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶'.includes(end)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if (L >= 1 && '的地得'.includes(end)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if (L === 1 && '了着过'.includes(word) && ['ul', 'uz', 'ug'].includes(pos)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if (L > 1 && '们子'.includes(end) && ['r', 'n'].includes(pos)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if (L > 1 && '上下'.includes(end) && ['s', 'l', 'f'].includes(pos)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if (L > 1 && '来去'.includes(end) && '上下进出回过起开'.includes(w[L - 2])) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  else if ((geIdx >= 1 && (isNumeric(w[geIdx - 1]) || '几有两半多各整每做是'.includes(w[geIdx - 1]))) || word === '个') finals[geIdx] = setTone(finals[geIdx], '5');
  else if (MUST_NEURAL.has(word) || MUST_NEURAL.has(end2)) finals[finals.length - 1] = setTone(finals[finals.length - 1], '5');
  const wl = splitWord(word), n0 = chars(wl[0]).length;
  const fl = [finals.slice(0, n0), finals.slice(n0)];
  wl.forEach((sw, i) => {
    if (MUST_NEURAL.has(sw) || MUST_NEURAL.has(chars(sw).slice(-2).join(''))) fl[i][fl[i].length - 1] = setTone(fl[i][fl[i].length - 1], '5');
  });
  return fl[0].concat(fl[1]);
}
function buSandhi(word, finals) {
  const w = chars(word);
  if (w.length === 3 && w[1] === '不') finals[1] = setTone(finals[1], '5');
  else w.forEach((c, i) => { if (c === '不' && i + 1 < w.length && last(finals[i + 1]) === '4') finals[i] = setTone(finals[i], '2'); });
  return finals;
}
function yiSandhi(word, finals) {
  const w = chars(word);
  if (w.includes('一') && w.filter(c => c !== '一').every(isNumeric)) return finals;
  if (w.length === 3 && w[1] === '一' && w[0] === w[2]) finals[1] = setTone(finals[1], '5');
  else if (word.startsWith('第一')) finals[1] = setTone(finals[1], '1');
  else w.forEach((c, i) => {
    if (c === '一' && i + 1 < w.length) {
      if (['4', '5'].includes(last(finals[i + 1]))) finals[i] = setTone(finals[i], '2');
      else if (!SANDHI_PUNC.includes(w[i + 1])) finals[i] = setTone(finals[i], '4');
    }
  });
  return finals;
}
function threeSandhi(word, finals) {
  const L = chars(word).length;
  if (L === 2 && allThree(finals)) finals[0] = setTone(finals[0], '2');
  else if (L === 3) {
    const wl = splitWord(word), n0 = chars(wl[0]).length;
    if (allThree(finals)) {
      if (n0 === 2) { finals[0] = setTone(finals[0], '2'); finals[1] = setTone(finals[1], '2'); }
      else if (n0 === 1) finals[1] = setTone(finals[1], '2');
    } else {
      const fl = [finals.slice(0, n0), finals.slice(n0)];
      fl.forEach((sub, i) => {
        if (allThree(sub) && sub.length === 2) fl[i][0] = setTone(fl[i][0], '2');
        else if (i === 1 && !allThree(sub) && last(fl[i][0]) === '3' && last(fl[0][fl[0].length - 1]) === '3')
          fl[0][fl[0].length - 1] = setTone(fl[0][fl[0].length - 1], '2');
        finals = fl[0].concat(fl[1]);
      });
    }
  } else if (L === 4) {
    const fl = [finals.slice(0, 2), finals.slice(2)];
    finals = [];
    for (const sub of fl) { if (allThree(sub)) sub[0] = setTone(sub[0], '2'); finals.push(...sub); }
  }
  return finals;
}
function mergeBu(seg) {
  const out = [];
  seg.forEach(([word, pos], i) => {
    if (!X_ENG.has(pos) && i > 0 && seg[i - 1][0] === '不') word = '不' + word;
    const nextPos = i + 1 < seg.length ? seg[i + 1][1] : null;
    if (word !== '不' || nextPos === null || X_ENG.has(nextPos)) out.push([word, pos]);
  });
  return out;
}
function mergeYi(seg) {
  let out = [], skip = false;
  seg.forEach(([word, pos], i) => {
    if (skip) { skip = false; return; }
    if (i >= 1 && word === '一' && i + 1 < seg.length && seg[i - 1][0] === seg[i + 1][0] && seg[i - 1][1] === 'v' && !X_ENG.has(seg[i + 1][1])) {
      out[out.length - 1] = [out[out.length - 1][0] + '一' + seg[i + 1][0], out[out.length - 1][1]]; skip = true;
    } else out.push([word, pos]);
  });
  seg = out; out = [];
  for (const [word, pos] of seg) {
    if (out.length && out[out.length - 1][0] === '一' && !X_ENG.has(pos)) out[out.length - 1] = [out[out.length - 1][0] + word, out[out.length - 1][1]];
    else out.push([word, pos]);
  }
  return out;
}
function mergeRedup(seg) {
  const out = [];
  for (const [word, pos] of seg) {
    if (out.length && word === out[out.length - 1][0] && !X_ENG.has(pos)) out[out.length - 1][0] += word;
    else out.push([word, pos]);
  }
  return out;
}
const isRedup = (w) => { const a = chars(w); return a.length === 2 && a[0] === a[1]; };
function finalsOf(word) {
  const f = lazyPinyin(word).map(x => x[1]);
  chars(word).forEach((c, i) => { if (c === '嗯') f[i] = 'n2'; });
  return f;
}
function mergeThree(seg, test) {
  const subs = seg.map(([w, p]) => X_ENG.has(p) ? ['0'] : finalsOf(w));
  const out = [], merged = seg.map(() => false);
  seg.forEach(([word, pos], i) => {
    if (!X_ENG.has(pos) && i >= 1 && test(subs[i - 1], subs[i]) && !merged[i - 1]) {
      if (!isRedup(seg[i - 1][0]) && chars(seg[i - 1][0]).length + chars(word).length <= 3) { out[out.length - 1][0] += word; merged[i] = true; }
      else out.push([word, pos]);
    } else out.push([word, pos]);
  });
  return out;
}
function mergeEr(seg) {
  const out = [];
  seg.forEach(([word, pos], i) => {
    if (i >= 1 && word === '儿' && !X_ENG.has(out[out.length - 1][1])) out[out.length - 1][0] += word;
    else out.push([word, pos]);
  });
  return out;
}
function preMerge(seg) {
  seg = mergeBu(seg); seg = mergeYi(seg); seg = mergeRedup(seg);
  seg = mergeThree(seg, (a, b) => allThree(a) && allThree(b));
  seg = mergeThree(seg, (a, b) => last(a[a.length - 1]) === '3' && last(b[0]) === '3');
  return mergeEr(seg);
}

// ---- misaki ZHFrontend ----------------------------------------------------------------------------
const ZH_MAP = { b: 'ㄅ', p: 'ㄆ', m: 'ㄇ', f: 'ㄈ', d: 'ㄉ', t: 'ㄊ', n: 'ㄋ', l: 'ㄌ', g: 'ㄍ', k: 'ㄎ', h: 'ㄏ', j: 'ㄐ', q: 'ㄑ', x: 'ㄒ', zh: 'ㄓ', ch: 'ㄔ', sh: 'ㄕ', r: 'ㄖ', z: 'ㄗ', c: 'ㄘ', s: 'ㄙ', a: 'ㄚ', o: 'ㄛ', e: 'ㄜ', ie: 'ㄝ', ai: 'ㄞ', ei: 'ㄟ', ao: 'ㄠ', ou: 'ㄡ', an: 'ㄢ', en: 'ㄣ', ang: 'ㄤ', eng: 'ㄥ', er: 'ㄦ', i: 'ㄧ', u: 'ㄨ', v: 'ㄩ', ii: 'ㄭ', iii: '十', ve: '月', ia: '压', ian: '言', iang: '阳', iao: '要', in: '阴', ing: '应', iong: '用', iou: '又', ong: '中', ua: '穵', uai: '外', uan: '万', uang: '王', uei: '为', uen: '文', ueng: '瓮', uo: '我', van: '元', vn: '云' };
for (const p of ';:,.!?/—…"()“” 12345R') ZH_MAP[p] = p;
const PUNC = new Set(';:,.!?—…"()“”');
const MUST_ERHUA = new Set(['小院儿', '胡同儿', '范儿', '老汉儿', '撒欢儿', '寻老礼儿', '妥妥儿', '媳妇儿']);
const NOT_ERHUA = new Set('虐儿 为儿 护儿 瞒儿 救儿 替儿 有儿 一儿 我儿 俺儿 妻儿 拐儿 聋儿 乞儿 患儿 幼儿 孤儿 婴儿 婴幼儿 连体儿 脑瘫儿 流浪儿 体弱儿 混血儿 蜜雪儿 舫儿 祖儿 美儿 应采儿 可儿 侄儿 孙儿 侄孙儿 女儿 男儿 红孩儿 花儿 虫儿 马儿 鸟儿 猪儿 猫儿 狗儿 少儿'.split(' '));
const UNK = '❓';

function initialsFinals(word) {
  const py = lazyPinyin(word), ini = [], fin = [];
  const f = py.map(x => x[1]);
  chars(word).forEach((c, i) => { if (c === '嗯') f[i] = 'n2'; });
  py.forEach(([c], i) => {
    let v = f[i];
    if (/^i\d/.test(v)) { if (['z', 'c', 's'].includes(c)) v = v.replaceAll('i', 'ii'); else if (['zh', 'ch', 'sh', 'r'].includes(c)) v = v.replaceAll('i', 'iii'); }
    ini.push(c); fin.push(v);
  });
  return [ini, fin];
}
function mergeErhua(ini, fin, word, pos) {
  const w = chars(word);
  fin.forEach((p, i) => { if (i === fin.length - 1 && w[i] === '儿' && p === 'er1') fin[i] = 'er2'; });
  if (!MUST_ERHUA.has(word) && (NOT_ERHUA.has(word) || ['a', 'j', 'nr'].includes(pos))) return [ini, fin];
  if (fin.length !== w.length) return [ini, fin];
  const ni = [], nf = [];
  fin.forEach((p, i) => {
    if (i === fin.length - 1 && w[i] === '儿' && (p === 'er2' || p === 'er5') && !NOT_ERHUA.has(w.slice(-2).join('')) && nf.length) {
      const l = nf[nf.length - 1]; nf[nf.length - 1] = l.slice(0, -1) + 'R' + l.slice(-1);
    } else { ni.push(ini[i]); nf.push(p); }
  });
  return [ni, nf];
}
function frontend(text) {
  const tokens = [];
  for (let [word, pos] of preMerge(psgCut(text))) {
    const a = chars(word);
    if (pos === 'x' && a.every(c => c >= '一' && c <= '鿿')) pos = 'X';
    else if (pos !== 'x' && PUNC.has(word)) pos = 'x';
    const tk = { tag: pos, ws: '', ph: null };
    if (X_ENG.has(pos)) {
      if (!isSpace(word)) { if (pos === 'x' && PUNC.has(word)) tk.ph = word; tokens.push(tk); }
      else if (tokens.length) tokens[tokens.length - 1].ws += word;
      continue;
    } else if (tokens.length && !X_ENG.has(tokens[tokens.length - 1].tag) && !tokens[tokens.length - 1].ws) tokens[tokens.length - 1].ws = '/';
    let [ini, fin] = initialsFinals(word);
    fin = threeSandhi(word, neuralSandhi(word, pos, yiSandhi(word, buSandhi(word, fin))));
    [ini, fin] = mergeErhua(ini, fin, word, pos);
    const phones = [];
    const n = Math.min(ini.length, fin.length);
    for (let i = 0; i < n; i++) {
      const c = ini[i], v = fin[i];
      if (c) phones.push(c);
      if (v && (!PUNC.has(v) || v !== c)) phones.push(v);
    }
    const ps = phones.join('_').replaceAll('_eR', '_er').replaceAll('R', '_R').replace(/(?=\d)/gu, '_').split('_');
    tk.ph = ps.map(p => ZH_MAP[p] ?? UNK).join('');
    tokens.push(tk);
  }
  return tokens.map(t => (t.ph ?? UNK) + t.ws).join('');
}

// ---- misaki ZHG2P(version='1.1') -------------------------------------------------------------------
// en: callback for embedded Latin words; misaki's default (no callback) drops them as unknown.
export function g2pZh11(text, en = null) {
  if (!text.trim()) return '';
  const t = mapPunctuation(transformAn2Cn(text));
  const segs = [];
  for (const m of t.matchAll(/([A-Za-z '-]*[A-Za-z][A-Za-z '-]*)|([^A-Za-z]+)/g)) {
    const e = (m[1] || '').trim(), z = (m[2] || '').trim();
    if (z) segs.push(frontend(z));
    else segs.push(en ? en(e) : UNK);
  }
  return segs.join(' ');
}

// ---- single syllables (pinyin-chart style drills) --------------------------------------------------
// Each syllable is read on its own, without tone sandhi, the way a pinyin chart pronounces it.
function finFix(c, v) {
  if (/^i\d/.test(v)) { if (['z', 'c', 's'].includes(c)) return v.replaceAll('i', 'ii'); if (['zh', 'ch', 'sh', 'r'].includes(c)) return v.replaceAll('i', 'iii'); }
  return v;
}
export function syllablePs(base, tone) {
  const e = SYLB[base]; if (!e) return null;
  const phones = [e[0], finFix(e[0], e[1] + tone)].filter(Boolean);
  const ps = phones.join('_').replaceAll('_eR', '_er').replaceAll('R', '_R').replace(/(?=\d)/gu, '_').split('_');
  return ps.map(p => ZH_MAP[p] ?? UNK).join('');
}
export const isSyllable = (base) => !!SYLB[base];
// Chinese characters -> [{base, tone}] using the same word-level pinyin readings as the sentence reader.
export function hanziSyllables(text) {
  const out = [];
  for (const run of text.match(/[\u3400-\u9fff]+/gu) || []) {
    for (const w of cut(run, true)) {
      for (const piece of mmseg(w)) {
        const raws = PH.has(piece) ? PH.get(piece) : chars(piece).map(c => CH.get(c) ?? null);
        chars(piece).forEach((c, i) => {
          if (c === '嗯') { out.push({ han: c, base: 'en', tone: '2' }); return; }
          const r = raws[i]; if (r == null) return;
          // 一 and 不 change tone in context; a drill reads their citation tone.
          const n = c === '一' ? 'yi1' : c === '不' ? 'bu4' : tone3(r).replace('ü', 'v'); out.push({ han: c, base: n.slice(0, -1), tone: n.slice(-1) });
        });
      }
    }
  }
  return out;
}
