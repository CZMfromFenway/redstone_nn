// Minimal, isolated test of the page's speckle() only. No other machinery.
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'index.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);/, '');   // do not auto-run

function makeEl() {
  return { children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c){this.children.push(c);return c;}, addEventListener(){}, removeEventListener(){},
    querySelectorAll(){return[];}, querySelector(){return null;}, classList:{add(){},remove(){},contains(){return false;}},
    getContext(){return {createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
    set innerHTML(v){this._h=v;}, get innerHTML(){return this._h;},
    set textContent(v){this._t=v;}, get textContent(){return this._t;}, width:8, height:8 };
}
global.document = { createElement: makeEl, getElementById: () => makeEl(), querySelectorAll: () => [], querySelector: () => makeEl() };
global.window = {};
global.fetch = async () => ({ ok: true, json: async () => ({}) });

new Function(code)();
const APP = global.window.APP;
const sp = APP.speckle;

function show(tag, img) {
  const s = sp(img);
  console.log(`  ${tag.padEnd(28)} agree=${s.agree.toFixed(4)}  iso=${s.iso}  blobs=${s.blobs}`);
}

console.log('speckle() on hand-made images (expected values in brackets):');
show('uniform, all level 7', new Uint8Array(64).fill(7));
show('uniform, all level 0', new Uint8Array(64));
const island = new Uint8Array(64).fill(3); island[4*8+4] = 9;
show('one island at (4,4)', island);
const chk = new Uint8Array(64); for (let r=0;r<8;r++) for (let c=0;c<8;c++) chk[r*8+c] = (r+c)%2 ? 0 : 15;
show('checkerboard', chk);
const half = new Uint8Array(64); for (let r=0;r<8;r++) for (let c=0;c<8;c++) half[r*8+c] = (r<4)?1:2;
show('top half 1 / bottom half 2', half);

console.log('\nexpected: uniform -> agree 1, iso 0, blobs 1');
console.log('expected: island -> agree<1, iso 1, blobs 2');
console.log('expected: checkerboard -> agree 0, iso 64, blobs 64');
console.log('expected: two bands -> agree high, iso 0, blobs 2');
console.log('SPECKLE_ISOLATED_TEST_DONE');
