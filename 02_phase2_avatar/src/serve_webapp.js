#!/usr/bin/env node
/*
 * Static file server for the experiment pages, PLUS a tiny label sink.
 *
 * POST /label  {model, pca:[22], grade:0..3, note}   -> appends one JSON line to
 *              results/preference/labels.jsonl
 *
 * The sink exists so the user's manual 优/良/中/差 judgements land on disk directly
 * instead of in browser storage that has to be exported by hand. Everything the
 * preference fitter needs is then already in one file.
 *
 * Usage:  node serve_webapp.js <directory> [port]
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = path.resolve(process.argv[2] || '.');
const port = parseInt(process.argv[3] || '8770', 10);
const repo = path.resolve(root, '..');
const LABELS = path.join(repo, 'results', 'preference', 'labels.jsonl');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
};

function readBody(req, limitBytes = 1 << 20) {
  return new Promise((resolve, reject) => {
    let n = 0; const chunks = [];
    req.on('data', c => {
      n += c.length;
      if (n > limitBytes) { reject(new Error('body too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];

  // ---- label sink -----------------------------------------------------------------
  if (req.method === 'POST' && url === '/label') {
    try {
      const body = JSON.parse(await readBody(req));
      if (!Array.isArray(body.pca) || body.pca.length !== 22)
        throw new Error('pca must be an array of 22 numbers');
      if (!Number.isInteger(body.grade) || body.grade < 0 || body.grade > 4)
        throw new Error('grade must be an integer 0..4');
      const rec = {
        ts: new Date().toISOString(),
        model: String(body.model || 'unknown'),
        grade: body.grade,
        pca: body.pca.map(Number),
        note: String(body.note || ''),
        // which grading scheme produced this label; classic and style must never be pooled
        scheme: String(body.scheme || 'classic'),
        // which labelling round this came from, so a new round never mixes with an
        // older one whose standard may have drifted
        round: String(body.round || ''),
        // echoed so the fitter can verify what the user actually saw
        predicted_levels: Array.isArray(body.predicted_levels) ? body.predicted_levels : null,
      };
      fs.mkdirSync(path.dirname(LABELS), { recursive: true });
      fs.appendFileSync(LABELS, JSON.stringify(rec) + '\n', 'utf8');
      const n = fs.readFileSync(LABELS, 'utf8').split('\n').filter(Boolean).length;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, total: n }));
      console.log(`label grade=${rec.grade} model=${rec.model} -> total ${n}`);
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: e.message }));
      console.log(`label rejected: ${e.message}`);
    }
    return;
  }

  // ---- label stats ----------------------------------------------------------------
  if (url.startsWith('/labelstats')) {
    let recs = [];
    try {
      recs = fs.readFileSync(LABELS, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
    } catch (e) { /* no file yet */ }
    // optional ?round= scoping: a new labelling round must not inherit older totals
    // url had its query string stripped at the top of the handler, so the round must
    // be read from req.url
    const rq = new URL('http://x' + req.url).searchParams.get('round');
    if (rq !== null && rq !== '') recs = recs.filter(r => String(r.round || '') === rq);
    else if (rq === '') recs = recs.filter(r => String(r.round || '') === '');
    const counts = [0, 0, 0, 0, 0];
    recs.forEach(r => { if (r.grade >= 0 && r.grade <= 4) counts[r.grade]++; });
    // PER-MODEL counts as well. The labelling page pools every record into one bar, which
    // made a fresh experiment look like it already had 428 answers; the ladder comparison
    // needs its own tally. `byModel[key]` is [差,中,良,优].
    const byModel = {};
    recs.forEach(r => {
      if (!(r.grade >= 0 && r.grade <= 4)) return;
      const k = String(r.model || 'unknown');
      if (!byModel[k]) byModel[k] = [0, 0, 0, 0, 0];
      byModel[k][r.grade]++;
    });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ total: recs.length, counts, byModel }));
    return;
  }

  // ---- static ---------------------------------------------------------------------
  let rel = decodeURIComponent(url);
  if (rel === '/') rel = '/index.html';
  const file = path.join(root, rel);
  if (!file.startsWith(root)) { res.writeHead(403); res.end('forbidden'); return; }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('not found: ' + rel);
      console.log(`404 ${rel}`);
      return;
    }
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
    console.log(`200 ${rel}  ${(data.length / 1e6).toFixed(2)} MB`);
  });
});

server.listen(port, '127.0.0.1', () => {
  console.log(`serving  ${root}`);
  console.log(`labels -> ${LABELS}`);
  console.log(`open:  http://127.0.0.1:${port}/`);
  console.log(`label tool:  http://127.0.0.1:${port}/label.html`);
  console.log('(Ctrl+C to stop)');
});
