// The Soundbench page's instruments, lifted out of Claude's drawing of a workbench.
// The drawing is one SVG whose instruments are groups with id inst-<id>; the page keeps four of them
// on the bench and stands the other three loose in the opening's empty corner.
import drawing from '../assets/soundbench-bench.svg?raw';

// The drawing's filter, hatching and ink, placed once on the page and shared by every piece.
export const defs = drawing.slice(drawing.indexOf('<defs>'), drawing.indexOf('</defs>') + '</defs>'.length);

const groups = Object.fromEntries(
  [...drawing.matchAll(/<g id="([\w-]+)"[^>]*>[\s\S]*?<\/g>/g)].map((m) => [m[1], m[0]]),
);

// Each instrument's public name, and what it plays when clicked:
// a few bars rendered by the library itself, served as /soundbench/<file>.mp3.
const instruments: Record<string, { file: string; name: string; plays: string }> = {
  'inst-bass': { file: 'bass', name: 'upright bass', plays: 'walking in D minor' },
  'inst-banjo': { file: 'banjo', name: 'banjo', plays: 'the one fitted to recordings below, opening a study of hammer-ons and slides' },
  'inst-piano': { file: 'piano', name: 'piano', plays: 'a small waltz in G' },
  'inst-guitar': { file: 'guitar', name: 'nylon-string guitar', plays: 'fingerpicked' },
  'inst-xeremia': { file: 'xeremia', name: 'xeremia', plays: 'a Mallorcan bagpipe, with a dance tune over its drone' },
  'inst-ximbomba': { file: 'ximbomba', name: 'ximbomba', plays: 'a Mallorcan friction drum, and real' },
  'inst-skinpipe': { file: 'skinpipe', name: 'skinpipe', plays: 'the one that was never built; a pipe whose bell is a drum skin' },
};

export const byFile = Object.fromEntries(Object.values(instruments).map((i) => [i.file, i]));

// A piece of the drawing: some of its groups, framed by a viewBox in the drawing's own coordinates.
const piece = (ids: string[], viewBox: string, turn = '') =>
  `<svg viewBox="${viewBox}"><g filter="url(#hand)"${turn && ` transform="${turn}"`}>${ids.map((id) => {
    const inst = instruments[id];
    return inst ? groups[id].replace(/<g id="[^"]+"[^>]*>/, `<g class="inst" data-file="${inst.file}">`) : groups[id];
  }).join('')}</g></svg>`;

export const bench = piece(['bench', 'inst-banjo', 'inst-piano', 'inst-guitar', 'inst-xeremia'], '142 36 1041 364');

// Loose pieces share one scale, so the instruments keep their drawn sizes relative to each other.
// turn redraws a piece turned within the drawing (its viewBox framing the turned piece);
// tilt and lift then set it down at a slant, a little above or below its neighbours.
const REM_PER_UNIT = 0.032;
const loosely = (id: string, viewBox: string, tilt: number, lift: number, turn = '') =>
  ({ svg: piece([id], viewBox, turn), tilt, lift, size: +(Number(viewBox.split(' ')[2]) * REM_PER_UNIT).toFixed(2) });

export const loose = [
  // the bass, laid down on its side with its scroll to the left
  loosely('inst-bass', '-107 151 382 116', -3, 0, 'rotate(-90 84 209)'),
  loosely('inst-ximbomba', '958 142 93 166', 6, 0.2),
  loosely('inst-skinpipe', '1042 142 155 172', -10, -0.8),
];
