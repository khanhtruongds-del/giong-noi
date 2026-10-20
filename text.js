// Sentence chunking shared by both engines (same rules as tts_app/text/frontend.py: sentence_chunks).
const SENTENCE_END = '.!?。！？…\n';
const SOFT_BREAK = ',，、;；:：';

export function sentenceChunks(text, limit = 200) {
  text = text.normalize('NFC');
  const chars = [...text];
  const sentences = [];
  let start = 0;
  chars.forEach((c, i) => {
    if (SENTENCE_END.includes(c) && (i + 1 === chars.length || !SENTENCE_END.includes(chars[i + 1]))) {
      sentences.push(chars.slice(start, i + 1)); start = i + 1;
    }
  });
  sentences.push(chars.slice(start));
  const pieces = [];
  for (let s of sentences) {
    while (s.length > limit) {
      const head = s.slice(0, limit);
      let cuts = head.map((c, i) => (SOFT_BREAK.includes(c) ? i + 1 : 0)).filter(Boolean);
      if (!cuts.length) cuts = head.map((c, i) => (/\s/.test(c) ? i + 1 : 0)).filter(Boolean);
      const cut = cuts.length && cuts[cuts.length - 1] > Math.floor(limit / 3) ? cuts[cuts.length - 1] : limit;
      pieces.push(s.slice(0, cut)); s = s.slice(cut);
    }
    pieces.push(s);
  }
  const chunks = [];
  let cur = [];
  for (const p of pieces) {
    if (cur.length && cur.length + p.length > limit) { chunks.push(cur.join('')); cur = []; }
    cur = cur.concat(p);
  }
  if (cur.length) chunks.push(cur.join(''));
  return chunks.map(c => c.split(/\s+/).filter(Boolean).join(' ')).filter(c => c.trim());
}
