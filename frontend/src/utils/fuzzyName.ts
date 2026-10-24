/**
 * Shared client-side OCR-aware fuzzy name matching utility.
 * Mirrors the backend `fuzzy_name_match_score` / `fuzzy_token_similarity` logic
 * in `backend/app/pipeline/lexicon.py` so client-side filtering and row highlighting
 * never discard or miss OCR-corrupted names returned by the backend.
 */

const FOLD_CACHE = new Map<string, string>();
const ORTH_CACHE = new Map<string, string>();
const EDIT_SIM_CACHE = new Map<string, number>();
const TOKEN_SIM_CACHE = new Map<string, number>();
const MAX_CACHE_SIZE = 16384;

export function foldText(text: string | null | undefined): string {
  if (!text) return "";
  const hit = FOLD_CACHE.get(text);
  if (hit !== undefined) return hit;
  const out = text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
  if (FOLD_CACHE.size >= MAX_CACHE_SIZE) {
    FOLD_CACHE.clear();
  }
  FOLD_CACHE.set(text, out);
  return out;
}

export function stripFemSuffix(word: string): string {
  let f = foldText(word).replace(/[^a-z]/g, "");
  while (f.length >= 7 && (f.endsWith("ovaova") || f.endsWith("oraova"))) {
    f = f.slice(0, -3);
  }
  if (f.length >= 5 && (f.endsWith("ova") || f.endsWith("ora") || f.endsWith("ura"))) {
    f = f.slice(0, -3);
  }
  if (f.length >= 5 && (f.endsWith("ska") || f.endsWith("cka"))) {
    return `${f.slice(0, -1)}y`;
  }
  if (f in ADJECTIVAL_STEMS) {
    return ADJECTIVAL_STEMS[f];
  }
  if (f.length >= 5 && f.slice(0, -1) in ADJECTIVAL_STEMS) {
    return ADJECTIVAL_STEMS[f.slice(0, -1)];
  }
  return f;
}

const ADJECTIVAL_STEMS: Record<string, string> = {
  stastn: "stastny",
  stasntn: "stastny",
  stasn: "stastny",
  siastn: "stastny",
  srastn: "stastny",
  rastn: "stastny",
  hastn: "stastny",
  hastu: "stastny",
  hastua: "stastny",
  novotn: "novotny",
  bystrick: "bystricky",
  majersk: "majersky",
};

const DIRECT_SURNAME_REPAIRS: Record<string, string> = {
  yargo: "varga",
  yarga: "varga",
  warga: "varga",
  vargo: "varga",
  farga: "varga",
  kowacs: "kovac",
  kowacz: "kovac",
  kovacs: "kovac",
  siastna: "stastny",
  siastny: "stastny",
  siastnova: "stastny",
  stasntna: "stastny",
  hastua: "stastny",
  hastuova: "stastny",
  stasnij: "stastny",
  stasny: "stastny",
};

const GIVEN_NAME_GROUPS: string[][] = [
  ["joannes", "johannes", "janos", "jan", "johann", "hans", "janko", "jano", "yoannes", "joarnes", "joames", "jannes"],
  ["stephanus", "istvan", "stefan", "stephan", "pista", "heter", "stephannus"],
  ["andreas", "andras", "ondrej", "andrej", "bandi"],
  ["josephus", "joseph", "jozsef", "jozef", "josef", "jozo", "doref", "joref"],
  ["michael", "mihaly", "michal", "miso"],
  ["paulus", "pal", "pavol", "pavel", "paul", "palo", "saulus", "gaulus"],
  ["petrus", "peter", "peto"],
  ["franciscus", "francis", "ferenc", "ferencz", "frantisek", "franz", "fero"],
  ["georgius", "gyorgy", "juraj", "duro", "georg", "jura"],
  ["martinus", "marton", "martin"],
  ["mathias", "matyas", "matthaeus", "mate", "matej", "matus"],
  ["nicolaus", "miklos", "mikulas", "nikolaus"],
  ["antonius", "anton", "antal", "tonko"],
  ["carolus", "karoly", "karol", "karel", "karl", "coxolus"],
  ["ladislaus", "laszlo", "ladislav", "laco"],
  ["thomas", "tamas", "tomas"],
  ["emericus", "emergicus", "imre", "imrich"],
  ["ludovicus", "lajos", "ludovit", "ludwig"],
  ["ignatius", "ignac", "ignaz"],
  ["adalbertus", "bela", "vojtech", "albertus", "albert"],
  ["alexander", "sandor"],
  ["sigismundus", "zsigmond", "zigmund", "sigmund"],
  ["julius", "gyula"],
  ["maria", "mara", "marika", "havia"],
  ["anna", "anca", "anka", "annuska", "ana"],
  ["catharina", "katalin", "katarina", "kata", "katka", "caharina"],
  ["elisabetha", "elizabetha", "elizabeth", "erzsebet", "alzbeta", "beta"],
  ["susanna", "zsuzsanna", "zuzana", "zuza"],
  ["veronica", "veronika", "vera", "vexonica"],
  ["magdalena", "magdolna", "magda", "haqdalena"],
  ["helena", "ilona", "elena", "lenka"],
  ["barbara", "borbala", "barbora"],
  ["margaritha", "margaretha", "margit", "margita", "gita"],
  ["theresia", "terez", "terezia"],
  ["rosalia", "rozalia", "roza"],
  ["sophia", "zsofia", "zofia"],
  ["julianna", "juliana", "julia", "juliska"],
  ["johanna", "jana", "janka", "jolan", "jolana"],
  ["ludovica", "lujza", "ludmila"],
  ["francisca", "ferike", "frantiska"],
  ["josepha", "jozefa", "jozefina"],
  ["irena", "iren", "irene", "steena", "stena"],
  ["vincencia", "vincentia", "vivencia"],
];

const GIVEN_TO_GROUP = new Map<string, number>();
GIVEN_NAME_GROUPS.forEach((grp, idx) => {
  for (const name of grp) {
    GIVEN_TO_GROUP.set(name, idx);
  }
});

export function orthographicFold(token: string | null | undefined): string {
  if (!token) return "";
  const hit = ORTH_CACHE.get(token);
  if (hit !== undefined) return hit;
  let s = foldText(token).replace(/[^a-z0-9]/g, "");
  if (!s) {
    ORTH_CACHE.set(token, "");
    return "";
  }
  const replacements: Array<[string, string]> = [
    ["sch", "s"],
    ["cz", "c"],
    ["cs", "c"],
    ["ts", "c"],
    ["tz", "c"],
    ["sz", "s"],
    ["zs", "z"],
    ["gy", "d"],
    ["dy", "d"],
    ["ny", "n"],
    ["nj", "n"],
    ["ly", "l"],
    ["lj", "l"],
    ["ph", "f"],
    ["th", "t"],
    ["gh", "g"],
    ["rn", "m"],
    ["w", "v"],
    ["y", "i"],
  ];
  for (const [src, dst] of replacements) {
    s = s.split(src).join(dst);
  }
  s = s.replace(/([a-z])\1+/g, "$1");
  if (ORTH_CACHE.size >= MAX_CACHE_SIZE) {
    ORTH_CACHE.clear();
  }
  ORTH_CACHE.set(token, s);
  return s;
}

const CONFUSION_PAIRS: Record<string, number> = {
  "v:y": 0.12,
  "s:y": 0.12,
  "j:y": 0.15,
  "i:y": 0.15,
  "g:y": 0.2,
  "q:y": 0.22,
  "j:q": 0.2,
  "a:o": 0.15,
  "n:u": 0.15,
  "u:v": 0.15,
  "r:u": 0.18,
  "n:v": 0.18,
  "m:n": 0.2,
  "m:u": 0.25,
  "m:w": 0.25,
  "h:m": 0.18,
  "c:e": 0.18,
  "c:o": 0.2,
  "e:o": 0.18,
  "c:t": 0.28,
  "c:k": 0.18,
  "g:k": 0.28,
  "n:r": 0.18,
  "l:t": 0.16,
  "i:l": 0.15,
  "f:l": 0.2,
  "j:l": 0.2,
  "i:t": 0.16,
  "f:i": 0.2,
  "i:j": 0.15,
  "j:t": 0.2,
  "f:j": 0.22,
  "g:q": 0.18,
  "g:j": 0.2,
  "r:v": 0.18,
  "v:w": 0.08,
  "f:v": 0.22,
  "p:s": 0.25,
  "g:s": 0.28,
  "j:s": 0.22,
  "f:s": 0.25,
  "h:s": 0.22,
  "f:t": 0.18,
  "d:t": 0.18,
  "b:p": 0.16,
  "b:k": 0.24,
  "h:k": 0.2,
  "b:h": 0.2,
  "d:o": 0.3,
  "s:z": 0.14,
  "c:z": 0.22,
  "r:z": 0.16,
};

function pairSubCost(a: string, b: string): number {
  if (a === b) return 0;
  const k = a < b ? `${a}:${b}` : `${b}:${a}`;
  return CONFUSION_PAIRS[k] ?? 0.95;
}

function ocrDpDistance(a: string, b: string): number {
  if (a === b) return 0;
  const n = a.length;
  const m = b.length;
  if (n === 0) return m * 0.78;
  if (m === 0) return n * 0.78;

  let prev = new Float64Array(m + 1);
  let curr = new Float64Array(m + 1);
  for (let j = 1; j <= m; j++) prev[j] = j * 0.78;

  for (let i = 1; i <= n; i++) {
    curr[0] = i * 0.78;
    const ca = a[i - 1];
    for (let j = 1; j <= m; j++) {
      const cb = b[j - 1];
      let sub = pairSubCost(ca, cb);
      if (sub >= 0.9 && i === 1 && j === 1) {
        sub = 1.25;
      }
      curr[j] = Math.min(
        prev[j] + 0.78,
        curr[j - 1] + 0.78,
        prev[j - 1] + sub
      );
    }
    const tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return prev[m];
}

export function weightedEditSimilarity(left: string, right: string): number {
  const aRaw = foldText(left).replace(/[^a-z0-9]/g, "");
  const bRaw = foldText(right).replace(/[^a-z0-9]/g, "");
  if (!aRaw || !bRaw) return 0;
  if (aRaw === bRaw) return 1.0;

  const cacheKey = aRaw < bRaw ? `${aRaw}|${bRaw}` : `${bRaw}|${aRaw}`;
  const hit = EDIT_SIM_CACHE.get(cacheKey);
  if (hit !== undefined) return hit;

  const aOrth = orthographicFold(left);
  const bOrth = orthographicFold(right);
  if (aOrth && aOrth === bOrth) {
    EDIT_SIM_CACHE.set(cacheKey, 0.97);
    return 0.97;
  }

  const simRaw = Math.max(
    0,
    1 - ocrDpDistance(aRaw, bRaw) / Math.max(aRaw.length, bRaw.length, 1)
  );
  const simOrth = Math.max(
    0,
    1 - ocrDpDistance(aOrth, bOrth) / Math.max(aOrth.length, bOrth.length, 1)
  );
  const out = Math.max(simRaw, simOrth);
  if (EDIT_SIM_CACHE.size >= MAX_CACHE_SIZE) {
    EDIT_SIM_CACHE.clear();
  }
  EDIT_SIM_CACHE.set(cacheKey, out);
  return out;
}

function surnameCoreStem(token: string): string {
  const f = foldText(token).replace(/[^a-z0-9]/g, "");
  if (f in DIRECT_SURNAME_REPAIRS) return DIRECT_SURNAME_REPAIRS[f];
  const stripped = stripFemSuffix(f);
  if (stripped in DIRECT_SURNAME_REPAIRS) return DIRECT_SURNAME_REPAIRS[stripped];
  return stripped;
}

export function fuzzyTokenSimilarity(qTok: string, candTok: string): number {
  const qf = foldText(qTok).replace(/[^a-z0-9]/g, "");
  const cf = foldText(candTok).replace(/[^a-z0-9]/g, "");
  if (!qf || !cf) return 0;
  if (qf === cf) return 1.0;
  if (qf.length >= 2 && cf.startsWith(qf)) return 0.95;
  if (qf.length >= 3 && cf.includes(qf)) return 0.9;

  if (/^\d+$/.test(qf) || /^\d+$/.test(cf)) return 0;

  const tokKey = `${qf}>${cf}`;
  const hit = TOKEN_SIM_CACHE.get(tokKey);
  if (hit !== undefined) return hit;

  let best = 0;
  const qGrp = GIVEN_TO_GROUP.get(qf);
  const cGrp = GIVEN_TO_GROUP.get(cf);
  if (qGrp !== undefined && qGrp === cGrp) {
    TOKEN_SIM_CACHE.set(tokKey, 0.96);
    return 0.96;
  }
  if (qGrp !== undefined) {
    for (const eq of GIVEN_NAME_GROUPS[qGrp]) {
      const sim = weightedEditSimilarity(eq, cf);
      if (sim >= 0.8) best = Math.max(best, 0.94 * sim);
    }
  } else if (cGrp !== undefined) {
    for (const eq of GIVEN_NAME_GROUPS[cGrp]) {
      const sim = weightedEditSimilarity(qf, eq);
      if (sim >= 0.8) best = Math.max(best, 0.94 * sim);
    }
  }

  const qStem = surnameCoreStem(qf);
  const cStem = surnameCoreStem(cf);
  if (qStem && cStem) {
    if (qStem === cStem) {
      TOKEN_SIM_CACHE.set(tokKey, 0.96);
      return 0.96;
    }
    const qOrth = orthographicFold(qStem);
    const cOrth = orthographicFold(cStem);
    if (qOrth && qOrth === cOrth) {
      TOKEN_SIM_CACHE.set(tokKey, 0.95);
      return 0.95;
    }

    best = Math.max(
      best,
      weightedEditSimilarity(qStem, cStem),
      weightedEditSimilarity(qf, cf)
    );
    const qRoot = qStem.length >= 5 ? qStem.replace(/[aoyi]+$/, "") : qStem;
    const cRoot = cStem.length >= 5 ? cStem.replace(/[aoyi]+$/, "") : cStem;
    if (qRoot.length >= 4 && cRoot.length >= 4) {
      best = Math.max(best, weightedEditSimilarity(qRoot, cRoot) * 0.97);
    }
  }

  if (TOKEN_SIM_CACHE.size >= MAX_CACHE_SIZE) {
    TOKEN_SIM_CACHE.clear();
  }
  TOKEN_SIM_CACHE.set(tokKey, best);
  return best;
}

const STOP_TOKENS = new Set([
  "rod",
  "rodena",
  "r",
  "nee",
  "geb",
  "szul",
  "ml",
  "st",
  "okr",
  "okres",
]);

export function matchesPersonNameQuery(
  parts: Array<string | null | undefined>,
  query: string,
  threshold = 0.76
): boolean {
  const qTrim = query.trim();
  if (!qTrim) return true;
  const qTokens = foldText(qTrim)
    .split(/[^a-z0-9]+/)
    .filter((t) => t && !STOP_TOKENS.has(t));
  if (qTokens.length === 0) return true;

  const candTokens: string[] = [];
  const fullParts: string[] = [];
  for (const p of parts) {
    if (!p) continue;
    const pf = foldText(p);
    if (!pf) continue;
    fullParts.push(pf);
    for (const w of pf.split(/[^a-z0-9]+/).filter(Boolean)) {
      if (!STOP_TOKENS.has(w)) {
        candTokens.push(w);
      }
    }
  }
  if (candTokens.length === 0) return false;

  const fullHaystack = fullParts.join(" ");
  const fullNeedle = qTokens.join(" ");
  if (fullNeedle && fullHaystack.includes(fullNeedle)) return true;

  return qTokens.every((qt) =>
    candTokens.some((ct) => fuzzyTokenSimilarity(qt, ct) >= threshold)
  );
}
