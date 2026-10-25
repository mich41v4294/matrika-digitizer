use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rusqlite::params_from_iter;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::gazetteer::{
    canonical_municipality_key_internal, normalize_house_number_internal,
};
use crate::{
    apply_digraphs_internal, clean_ascii_alnum, fold_text_diacritics_internal, open_sqlite,
    orthographic_fold_internal, phonetic_skeleton_internal, round4,
    weighted_edit_similarity_internal,
};

static GIVEN_GROUPS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "JOANNES",
        vec![
            "joannes", "johannes", "jános", "janos", "ján", "jan", "johann", "hans", "janko",
            "jano", "yoannes",
        ],
    );
    m.insert(
        "STEPHANUS",
        vec![
            "stephanus", "istván", "istvan", "štefan", "stefan", "stephan", "pišta", "pista",
        ],
    );
    m.insert(
        "ANDREAS",
        vec![
            "andreas", "andrás", "andras", "ondrej", "andrej", "bandi", "endrik",
        ],
    );
    m.insert(
        "JOSEPHUS",
        vec![
            "josephus", "joseph", "józsef", "jozsef", "jozef", "josef", "jožo", "jozo",
        ],
    );
    m.insert(
        "MICHAEL",
        vec![
            "michael", "michaël", "mihály", "mihaly", "michal", "mišo", "miso",
        ],
    );
    m.insert(
        "PAULUS",
        vec!["paulus", "pál", "pal", "pavol", "pavel", "paul", "paľo", "palo"],
    );
    m.insert("PETRUS", vec!["petrus", "péter", "peter", "peťo", "peto"]);
    m.insert(
        "FRANCISCUS",
        vec![
            "franciscus", "francis", "ferenc", "ferencz", "františek", "frantisek", "franz", "fero",
        ],
    );
    m.insert(
        "GEORGIUS",
        vec![
            "georgius", "györgy", "gyorgy", "juraj", "ďuro", "duro", "georg", "jura",
        ],
    );
    m.insert("MARTINUS", vec!["martinus", "márton", "marton", "martin"]);
    m.insert(
        "MATHIAS",
        vec![
            "mathias", "mátyás", "matyas", "matthaeus", "máté", "matej", "matúš", "matus",
        ],
    );
    m.insert(
        "NICOLAUS",
        vec![
            "nicolaus", "miklós", "miklos", "mikuláš", "mikulas", "nikolaus",
        ],
    );
    m.insert("ANTONIUS", vec!["antonius", "anton", "antal", "tonko"]);
    m.insert(
        "CAROLUS",
        vec!["carolus", "károly", "karoly", "karol", "karel", "karl"],
    );
    m.insert(
        "LADISLAUS",
        vec!["ladislaus", "lászló", "laszlo", "ladislav", "laco"],
    );
    m.insert("THOMAS", vec!["thomas", "tamás", "tamas", "tomáš", "tomas"]);
    m.insert("EMERICUS", vec!["emericus", "emergicus", "imre", "imrich"]);
    m.insert(
        "LUDOVICUS",
        vec!["ludovicus", "lajos", "ľudovít", "ludovit", "ludwig"],
    );
    m.insert("IGNATIUS", vec!["ignatius", "ignác", "ignac", "ignaz"]);
    m.insert(
        "ADALBERTUS",
        vec!["adalbertus", "béla", "bela", "vojtech", "albertus", "albert"],
    );
    m.insert(
        "ALEXANDER",
        vec!["alexander", "sándor", "sandor", "šándor"],
    );
    m.insert(
        "SIGISMUNDUS",
        vec!["sigismundus", "zsigmond", "žigmund", "zigmund", "sigmund"],
    );
    m.insert("JULIUS", vec!["julius", "gyula", "július"]);
    m.insert("MARIA", vec!["maria", "mária", "mara", "marika"]);
    m.insert("ANNA", vec!["anna", "anca", "anka", "annuška", "ana"]);
    m.insert(
        "CATHARINA",
        vec![
            "catharina", "katalin", "katarína", "katarina", "kata", "katka",
        ],
    );
    m.insert(
        "ELISABETHA",
        vec![
            "elisabetha",
            "elizabetha",
            "elizabeth",
            "erzsébet",
            "erzsebet",
            "alžbeta",
            "alzbeta",
            "beta",
        ],
    );
    m.insert("SUSANNA", vec!["susanna", "zsuzsanna", "zuzana", "zuza"]);
    m.insert("VERONICA", vec!["veronica", "veronika", "vera"]);
    m.insert(
        "MAGDALENA",
        vec!["magdalena", "magdolna", "magdaléna", "magda"],
    );
    m.insert("HELENA", vec!["helena", "ilona", "elena", "lenka"]);
    m.insert("EVA", vec!["eva", "éva"]);
    m.insert("BARBARA", vec!["barbara", "borbála", "borbala", "barbora"]);
    m.insert(
        "MARGARETA",
        vec!["margaritha", "margaretha", "margit", "margita", "gita"],
    );
    m.insert(
        "THERESIA",
        vec!["theresia", "teréz", "terez", "terézia", "terezia"],
    );
    m.insert("ROSALIA", vec!["rosalia", "rozália", "rozalia", "róza"]);
    m.insert("SOPHIA", vec!["sophia", "zsófia", "zsofia", "žofia", "zofia"]);
    m.insert(
        "JULIANNA",
        vec!["julianna", "juliana", "júlia", "julia", "juliska"],
    );
    m.insert(
        "JOHANNA",
        vec!["johanna", "jana", "janka", "jolan", "jolana"],
    );
    m.insert(
        "LUDOVICA",
        vec!["ludovica", "lujza", "ľudmila", "ludmila"],
    );
    m.insert(
        "FRANCISCA",
        vec!["francisca", "ferike", "františka", "frantiska"],
    );
    m.insert(
        "JOSEPHA",
        vec!["josepha", "jozefa", "jozefína", "jozefina"],
    );
    m.insert("IRENA", vec!["irena", "irén", "irene"]);
    m.insert("VINCENCIA", vec!["vincencia", "vincentia"]);
    m
});

static GIVEN_TO_CID: Lazy<HashMap<String, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (&cid, variants) in GIVEN_GROUPS.iter() {
        for &v in variants {
            m.insert(v.to_string(), cid);
            m.insert(fold_text_diacritics_internal(v), cid);
        }
    }
    let repairs = [
        ("joarnes", "joannes"),
        ("joames", "joannes"),
        ("jannes", "joannes"),
        ("seanes", "joannes"),
        ("svannes", "joannes"),
        ("joanner", "joannes"),
        ("stephannus", "stephanus"),
        ("stephans", "stephanus"),
        ("seghames", "stephanus"),
        ("vexonica", "veronica"),
        ("caharina", "catharina"),
        ("catharin", "catharina"),
        ("matheng", "matthaeus"),
        ("coxolus", "carolus"),
        ("heter", "štefan"),
        ("steena", "irena"),
        ("stena", "irena"),
        ("vivencia", "vincencia"),
        ("havia", "mária"),
        ("haqdalena", "magdaléna"),
        ("doref", "jozef"),
        ("joref", "jozef"),
    ];
    for (ocr_g, rep_g) in repairs {
        if let Some(&cid) = m.get(rep_g) {
            m.insert(ocr_g.to_string(), cid);
            m.insert(fold_text_diacritics_internal(ocr_g), cid);
        }
    }
    m
});

static ADJECTIVAL_STEMS_FOLDED: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (k, v) in [
        ("stastn", "stastny"),
        ("stasntn", "stastny"),
        ("stasn", "stastny"),
        ("siastn", "stastny"),
        ("srastn", "stastny"),
        ("rastn", "stastny"),
        ("hastn", "stastny"),
        ("hastu", "stastny"),
        ("opatrn", "opatrny"),
        ("opacit", "opacity"),
        ("novotn", "novotny"),
        ("vesel", "vesely"),
        ("pokorn", "pokorny"),
        ("ciern", "cierny"),
        ("cern", "cerny"),
        ("biel", "biely"),
        ("mal", "maly"),
        ("nov", "novy"),
        ("star", "stary"),
        ("dlh", "dlhy"),
        ("kratk", "kratky"),
        ("such", "suchy"),
        ("tich", "tichy"),
        ("mudr", "mudry"),
        ("pekn", "pekny"),
        ("krasn", "krasny"),
        ("konecn", "konecny"),
        ("slobodn", "slobodny"),
        ("dobr", "dobry"),
        ("hrd", "hrdy"),
        ("siv", "sivy"),
        ("piln", "pilny"),
        ("zlat", "zlaty"),
        ("zlt", "zlty"),
        ("bohat", "bohaty"),
        ("chud", "chudy"),
        ("mlad", "mlady"),
        ("tvrd", "tvrdy"),
        ("hluch", "hluchy"),
        ("plach", "plachy"),
        ("sladk", "sladky"),
        ("hladk", "hladky"),
        ("prudk", "prudky"),
        ("tazk", "tazky"),
        ("vysok", "vysoky"),
        ("sirok", "siroky"),
        ("hlbok", "hlboky"),
        ("velk", "velky"),
        ("tenk", "tenky"),
        ("makk", "makky"),
        ("chrom", "chromy"),
        ("slep", "slepy"),
        ("slab", "slaby"),
        ("hrub", "hruby"),
        ("rysav", "rysavy"),
        ("smel", "smely"),
        ("zrel", "zrely"),
        ("ostr", "ostry"),
        ("bystr", "bystry"),
        ("mokr", "mokry"),
        ("lav", "lavy"),
        ("prav", "pravy"),
        ("bystrick", "bystricky"),
        ("majersk", "majersky"),
        ("brestovansk", "brestovansky"),
        ("bukovsk", "bukovsky"),
        ("benkovsk", "benkovsky"),
        ("komarnansk", "komarnansky"),
        ("bohumick", "bohumicky"),
        ("bohunick", "bohunicky"),
        ("panovsk", "panovsky"),
    ] {
        m.insert(k, v);
    }
    m
});

static SURNAME_STEM_A: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "paraba", "varga", "homola", "vacula", "skoda", "barta", "mikula", "havalda", "kucera",
        "chalupa", "sykora", "prochazka", "zemlicka", "hruska", "krupka", "buchta", "liska",
    ]
    .into_iter()
    .collect()
});

static SURNAME_STEM_O: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["bartko", "szabo", "marko", "simko", "vanco"]
        .into_iter()
        .collect()
});

static KNOWN_SURNAME_FOLDED: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "kral",
        "bystricky",
        "paraba",
        "koval",
        "kabat",
        "sidlik",
        "varga",
        "horvath",
        "horvat",
        "kovac",
        "kovacik",
        "toth",
        "ciz",
        "mikus",
        "brestovansky",
        "polak",
        "sedlak",
        "krajcovic",
        "bartko",
        "novak",
        "balaz",
        "hudec",
        "bielik",
        "kralik",
        "moravec",
        "holik",
        "holic",
        "homola",
        "vacula",
        "bodi",
        "skoda",
        "stastny",
        "opatrny",
        "sulak",
        "galik",
        "barta",
        "mikula",
        "havalda",
        "kucera",
        "chalupa",
        "sykora",
        "prochazka",
        "zemlicka",
        "hruska",
        "krupka",
        "chovanec",
        "lukac",
        "molnar",
        "nagy",
        "szabo",
        "nemec",
        "kollar",
        "oravec",
        "sevcik",
        "urban",
        "marko",
        "kubik",
        "tomasik",
        "pavlik",
        "simko",
        "zeman",
        "rybar",
        "slezak",
        "valach",
        "blazek",
        "masar",
        "kadlec",
        "kovar",
        "buchta",
        "liska",
        "vanco",
        "majersky",
        "bukovsky",
        "benkovsky",
        "komarnansky",
        "bohumicky",
        "pilny",
        "harsanyi",
        "vinter",
        "habel",
        "danihel",
        "fabian",
        "mihalik",
        "gabor",
        "sochor",
        "duris",
        "vrazic",
        "nahlik",
        "holomek",
        "zobek",
        "jobek",
    ]
    .into_iter()
    .collect()
});

static TOKEN_SIM_CACHE: Lazy<Mutex<HashMap<(String, String), f64>>> =
    Lazy::new(|| Mutex::new(HashMap::with_capacity(16384)));

pub fn resolve_given_cid_pub(token: &str) -> Option<String> {
    resolve_given_cid_internal(token)
}

fn resolve_given_cid_internal(token: &str) -> Option<String> {
    let first_word = token.split_whitespace().next().unwrap_or("");
    let mut cleaned = String::with_capacity(first_word.len());
    for c in first_word.chars() {
        if c.is_alphabetic() {
            for lc in c.to_lowercase() {
                cleaned.push(lc);
            }
        }
    }
    if cleaned.is_empty() {
        return None;
    }
    if let Some(&cid) = GIVEN_TO_CID.get(&cleaned) {
        return Some(cid.to_string());
    }
    let folded = fold_text_diacritics_internal(&cleaned);
    if let Some(&cid) = GIVEN_TO_CID.get(&folded) {
        return Some(cid.to_string());
    }
    Some(folded.to_ascii_uppercase())
}

pub fn surname_core_stem_internal(token: &str) -> String {
    let mut s = token
        .trim_matches(|c: char| {
            matches!(
                c,
                ' ' | '.' | ',' | ';' | ':' | '(' | ')' | '/' | '\\' | '\'' | '’' | '`' | '"' | '„'
                    | '”' | '-'
            )
        })
        .to_string();
    if s.is_empty() {
        return String::new();
    }
    for (src, dst) in [
        ("Ś", "Š"),
        ("ś", "š"),
        ("Ć", "Č"),
        ("ć", "č"),
        ("Đ", "D"),
        ("đ", "d"),
        ("Ł", "L"),
        ("ł", "l"),
    ] {
        if s.contains(src) {
            s = s.replace(src, dst);
        }
    }

    let low = s.to_lowercase();
    let repaired = match low.as_str() {
        "yargo" | "yarga" | "warga" | "vargo" | "farga" => "Varga",
        "kovaasik" | "kowacsik" => "Kovacsik",
        "kowacs" | "kovacs" => "Kovács",
        "terenesik" => "Terencsík",
        "trencsik" => "Trencsík",
        "stasntna" | "stastnaova" | "stasntnaova" | "stastnyova" | "šťastnaová" | "šťastnáová"
        | "šťastnýová" | "siastna" | "siastnová" | "siastnova" | "hastua" | "hastuová"
        | "haštuová" | "řastná" | "rastna" | "srastna" | "stasnij" | "stasny" => "stastny",
        "buchta-ová" | "buchta-ova" | "buchstova" | "buchst" | "buchlova" | "buchlov"
        | "buchtova" | "buchta" => "buchta",
        "vančoová" | "vancoova" | "vancouver" | "vancouverova" | "vancora" | "vaniora"
        | "vanicova" | "vanic" | "vanicu" | "vamico" | "vanco" | "vancova" => "vanco",
        _ => "",
    };
    if !repaired.is_empty() {
        return clean_ascii_alnum(repaired);
    }

    let mut low_f = clean_ascii_alnum(&s);
    if low_f.is_empty() {
        return low_f;
    }

    while (low_f.ends_with("ovaova")) && low_f.len() >= 7 {
        low_f.truncate(low_f.len() - 3);
    }

    if matches!(
        low_f.as_str(),
        "stasny"
            | "stasnij"
            | "stastny"
            | "stastna"
            | "stasntna"
            | "siastna"
            | "siastnova"
            | "srastna"
            | "srastny"
            | "rastna"
            | "hastua"
            | "hastuova"
            | "hastu"
    ) {
        return "stastny".to_string();
    }

    if low_f.ends_with("aura") && low_f.len() >= 6 {
        let cand_an = format!("{}an", &low_f[..low_f.len() - 4]);
        if KNOWN_SURNAME_FOLDED.contains(cand_an.as_str()) {
            return cand_an;
        }
    }
    for nh_suf in ["nhova", "nha", "nh"] {
        if low_f.ends_with(nh_suf) && low_f.len() >= 5 {
            let prefix = &low_f[..low_f.len() - nh_suf.len()];
            let cand_nsky = format!("{}nsky", prefix);
            if KNOWN_SURNAME_FOLDED.contains(cand_nsky.as_str()) {
                return cand_nsky;
            }
        }
    }

    for suf in ["ovra", "ova", "ove", "ora", "ura"] {
        if low_f.ends_with(suf)
            && low_f.len() > suf.len() + 2
            && !low_f.ends_with("bora")
            && !low_f.ends_with("dora")
            && !low_f.ends_with("kora")
        {
            if (suf == "ora" || suf == "ura") && SURNAME_STEM_A.contains(low_f.as_str()) {
                return low_f;
            }
            let base_f = &low_f[..low_f.len() - suf.len()];
            if let Some(&adj_masc) = ADJECTIVAL_STEMS_FOLDED.get(base_f) {
                return adj_masc.to_string();
            }
            if base_f.len() >= 4
                && (base_f.ends_with('y') || base_f.ends_with('a') || base_f.ends_with('i'))
            {
                if let Some(&adj_masc) = ADJECTIVAL_STEMS_FOLDED.get(&base_f[..base_f.len() - 1]) {
                    return adj_masc.to_string();
                }
            }
            if (base_f.ends_with("ska") || base_f.ends_with("sky")) && base_f.len() >= 5 {
                return format!("{}sky", &base_f[..base_f.len() - 3]);
            }
            if (base_f.ends_with("cka") || base_f.ends_with("cky")) && base_f.len() >= 5 {
                return format!("{}cky", &base_f[..base_f.len() - 3]);
            }
            let cand_a = format!("{}a", base_f);
            if SURNAME_STEM_A.contains(cand_a.as_str()) {
                return cand_a;
            }
            let cand_o = format!("{}o", base_f);
            if SURNAME_STEM_O.contains(cand_o.as_str()) {
                return cand_o;
            }
            return base_f.to_string();
        }
    }

    if let Some(&adj_masc) = ADJECTIVAL_STEMS_FOLDED.get(low_f.as_str()) {
        return adj_masc.to_string();
    }
    if low_f.len() >= 4
        && (low_f.ends_with('y') || low_f.ends_with('a') || low_f.ends_with('i'))
    {
        if let Some(&adj_masc) = ADJECTIVAL_STEMS_FOLDED.get(&low_f[..low_f.len() - 1]) {
            return adj_masc.to_string();
        }
    }
    if low_f.ends_with("ska") && low_f.len() >= 5 {
        return format!("{}sky", &low_f[..low_f.len() - 3]);
    }
    if low_f.ends_with("cka") && low_f.len() >= 5 {
        return format!("{}cky", &low_f[..low_f.len() - 3]);
    }
    let low_orig = s.to_lowercase();
    for fem_acc in [
        "ská", "cká", "ná", "tá", "dá", "lá", "rá", "vá", "má", "pá", "bá", "há", "chá", "žá",
        "šá", "čá", "ká",
    ] {
        if low_orig.ends_with(fem_acc)
            && !low_orig.ends_with("ová")
            && !low_orig.ends_with("orá")
            && !low_orig.ends_with("urá")
            && low_f.ends_with('a')
        {
            return format!("{}y", &low_f[..low_f.len() - 1]);
        }
    }

    low_f
}

pub fn fuzzy_token_similarity_internal(q_tok: &str, cand_tok: &str) -> f64 {
    if q_tok.is_empty() || cand_tok.is_empty() {
        return 0.0;
    }
    let q_f = clean_ascii_alnum(q_tok);
    let c_f = clean_ascii_alnum(cand_tok);
    if q_f.is_empty() || c_f.is_empty() {
        return 0.0;
    }
    if q_f == c_f {
        return 1.0;
    }
    if q_f.len() >= 2 && c_f.starts_with(&q_f) {
        return 0.95;
    }
    if q_f.len() >= 3 && c_f.contains(&q_f) {
        return 0.90;
    }
    if q_f.bytes().all(|b| b.is_ascii_digit()) || c_f.bytes().all(|b| b.is_ascii_digit()) {
        return 0.0;
    }

    let cache_key = (q_tok.to_string(), cand_tok.to_string());
    if let Ok(guard) = TOKEN_SIM_CACHE.lock() {
        if let Some(&cached) = guard.get(&cache_key) {
            return cached;
        }
    }

    let mut best = 0.0_f64;

    let q_cid = resolve_given_cid_internal(q_tok);
    let c_cid = resolve_given_cid_internal(cand_tok);
    if let Some(ref q_c) = q_cid {
        if let Some(eqs) = GIVEN_GROUPS.get(q_c.as_str()) {
            if c_cid.as_deref() == Some(q_c.as_str()) {
                if let Ok(mut guard) = TOKEN_SIM_CACHE.lock() {
                    if guard.len() > 65536 {
                        guard.clear();
                    }
                    guard.insert(cache_key, 0.96);
                }
                return 0.96;
            }
            for &eq in eqs {
                let eq_sim = weighted_edit_similarity_internal(eq, cand_tok);
                if eq_sim >= 0.80 {
                    best = best.max(round4(0.94 * eq_sim));
                }
            }
        } else if let Some(ref c_c) = c_cid {
            if let Some(eqs) = GIVEN_GROUPS.get(c_c.as_str()) {
                for &eq in eqs {
                    let eq_sim = weighted_edit_similarity_internal(q_tok, eq);
                    if eq_sim >= 0.80 {
                        best = best.max(round4(0.94 * eq_sim));
                    }
                }
            }
        }
    }

    let q_stem = surname_core_stem_internal(q_tok);
    let c_stem = surname_core_stem_internal(cand_tok);
    if !q_stem.is_empty() && !c_stem.is_empty() {
        if q_stem == c_stem {
            if let Ok(mut guard) = TOKEN_SIM_CACHE.lock() {
                if guard.len() > 65536 {
                    guard.clear();
                }
                guard.insert(cache_key, 0.96);
            }
            return 0.96;
        }
        let q_orth = orthographic_fold_internal(&q_stem);
        let c_orth = orthographic_fold_internal(&c_stem);
        if !q_orth.is_empty() && q_orth == c_orth {
            if let Ok(mut guard) = TOKEN_SIM_CACHE.lock() {
                if guard.len() > 65536 {
                    guard.clear();
                }
                guard.insert(cache_key, 0.95);
            }
            return 0.95;
        }
        if q_orth.len() >= 3 && (c_orth.starts_with(&q_orth) || q_orth.starts_with(&c_orth)) {
            let min_l = q_orth.len().min(c_orth.len()) as f64;
            let max_l = q_orth.len().max(c_orth.len()) as f64;
            if (min_l / max_l) >= 0.75 {
                best = best.max(0.91);
            }
        }

        best = best
            .max(weighted_edit_similarity_internal(&q_stem, &c_stem))
            .max(weighted_edit_similarity_internal(q_tok, cand_tok));

        let q_root = if q_stem.len() >= 5 {
            q_stem.trim_end_matches(['a', 'o', 'y', 'i'])
        } else {
            q_stem.as_str()
        };
        let c_root = if c_stem.len() >= 5 {
            c_stem.trim_end_matches(['a', 'o', 'y', 'i'])
        } else {
            c_stem.as_str()
        };
        if q_root.len() >= 4
            && c_root.len() >= 4
            && (q_root != q_stem.as_str() || c_root != c_stem.as_str())
        {
            let r_sim = weighted_edit_similarity_internal(q_root, c_root);
            best = best.max(round4(r_sim * 0.97));
        }
    }

    let res = round4(best);
    if let Ok(mut guard) = TOKEN_SIM_CACHE.lock() {
        if guard.len() > 65536 {
            guard.clear();
        }
        guard.insert(cache_key, res);
    }
    res
}

fn is_query_stop_token(folded: &str) -> bool {
    matches!(
        folded,
        "rod" | "rodena" | "r" | "nee" | "geb" | "szul" | "ml" | "st" | "okr" | "okres"
    )
}

fn is_name_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            'á' | 'é'
                | 'í'
                | 'ó'
                | 'ö'
                | 'ő'
                | 'ú'
                | 'ü'
                | 'ű'
                | 'Á'
                | 'É'
                | 'Í'
                | 'Ó'
                | 'Ö'
                | 'Ő'
                | 'Ú'
                | 'Ü'
                | 'Ű'
                | 'ä'
                | 'ë'
                | 'Ä'
                | 'Ë'
                | 'č'
                | 'ď'
                | 'ľ'
                | 'ĺ'
                | 'ň'
                | 'ŕ'
                | 'ř'
                | 'š'
                | 'ť'
                | 'ž'
                | 'Č'
                | 'Ď'
                | 'Ľ'
                | 'Ĺ'
                | 'Ň'
                | 'Ŕ'
                | 'Ř'
                | 'Š'
                | 'Ť'
                | 'Ž'
                | 'ý'
                | 'Ý'
                | 'ô'
                | 'Ô'
        )
}

fn extract_name_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if is_name_token_char(ch) {
            cur.push(ch);
        } else if !cur.is_empty() {
            let folded = fold_text_diacritics_internal(&cur);
            if !is_query_stop_token(&folded) {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        let folded = fold_text_diacritics_internal(&cur);
        if !is_query_stop_token(&folded) {
            out.push(cur);
        }
    }
    out
}

pub fn fuzzy_name_match_score_internal(
    query_text: &str,
    candidate_texts: &[Option<String>],
    threshold: f64,
) -> f64 {
    let q_trim = query_text.trim();
    if q_trim.is_empty() {
        return 1.0;
    }
    let q_tokens = extract_name_tokens(q_trim);
    if q_tokens.is_empty() {
        return 1.0;
    }

    let mut cand_tokens: Vec<String> = Vec::new();
    let mut seen_cand: HashSet<String> = HashSet::new();
    let mut full_folded_parts: Vec<String> = Vec::new();

    for opt_t in candidate_texts {
        let s = match opt_t {
            Some(ref s) => s.trim(),
            None => continue,
        };
        if s.is_empty() {
            continue;
        }
        full_folded_parts.push(fold_text_diacritics_internal(s));
        for tok in extract_name_tokens(s) {
            if seen_cand.insert(tok.clone()) {
                cand_tokens.push(tok);
            }
        }
    }

    if cand_tokens.is_empty() {
        return 0.0;
    }

    let q_full_folded = fold_text_diacritics_internal(&q_tokens.join(" "));
    let full_haystack = full_folded_parts.join(" ");
    if !q_full_folded.is_empty() && full_haystack.contains(&q_full_folded) {
        return if q_full_folded == full_haystack.trim() {
            1.0
        } else {
            0.98
        };
    }

    let mut sum_scores = 0.0_f64;
    for qt in &q_tokens {
        let mut best_for_qt = 0.0_f64;
        for ct in &cand_tokens {
            let sim = fuzzy_token_similarity_internal(qt, ct);
            if sim > best_for_qt {
                best_for_qt = sim;
                if best_for_qt >= 0.99 {
                    break;
                }
            }
        }
        if best_for_qt < threshold {
            return 0.0;
        }
        sum_scores += best_for_qt;
    }

    round4(sum_scores / (q_tokens.len() as f64))
}

#[pyfunction]
#[pyo3(signature = (given_name=None))]
pub fn resolve_given_name_canonical_id_rs(given_name: Option<&str>) -> Option<String> {
    given_name.and_then(resolve_given_cid_internal)
}

#[pyfunction]
#[pyo3(signature = (token=None))]
pub fn surname_core_stem_rs(token: Option<&str>) -> String {
    match token {
        Some(s) => surname_core_stem_internal(s),
        None => String::new(),
    }
}

#[pyfunction]
#[pyo3(signature = (q_tok=None, cand_tok=None))]
pub fn fuzzy_token_similarity_rs(q_tok: Option<&str>, cand_tok: Option<&str>) -> f64 {
    match (q_tok, cand_tok) {
        (Some(q), Some(c)) => fuzzy_token_similarity_internal(q, c),
        _ => 0.0,
    }
}

#[pyfunction]
#[pyo3(signature = (query_text, candidate_texts, threshold=0.76))]
pub fn fuzzy_name_match_score_rs(
    query_text: Option<&str>,
    candidate_texts: Vec<Option<String>>,
    threshold: f64,
) -> f64 {
    match query_text {
        Some(q) => fuzzy_name_match_score_internal(q, &candidate_texts, threshold),
        None => 1.0,
    }
}

struct SearchableFactRow {
    id: i64,
    census_sheet_id: Option<i64>,
    given_name_raw: Option<String>,
    given_name_sk: Option<String>,
    given_name_hu: Option<String>,
    surname_raw: Option<String>,
    surname_normalized: Option<String>,
    mother_maiden_surname: Option<String>,
    father_name: Option<String>,
    mother_name: Option<String>,
    spouse_name: Option<String>,
    surname_candidates_json: Option<String>,
    village_raw: Option<String>,
    village_sk: Option<String>,
    village_hu: Option<String>,
    fs_place_standardized: Option<String>,
    birth_place_raw: Option<String>,
    birth_place_sk: Option<String>,
    birth_place_municipality: Option<String>,
    home_jurisdiction: Option<String>,
    social_status_or_occupation: Option<String>,
    occupation_position: Option<String>,
    religion: Option<String>,
    observationes_raw: Option<String>,
    house_number: Option<String>,
    event_year: Option<i64>,
    birth_year_est: Option<i64>,
    ocr_txt: Option<String>,
}

fn extract_struct_name_cands(f: &SearchableFactRow) -> Vec<String> {
    let mut parts: Vec<String> = Vec::with_capacity(12);
    for val in [
        &f.given_name_raw,
        &f.given_name_sk,
        &f.given_name_hu,
        &f.surname_raw,
        &f.surname_normalized,
        &f.mother_maiden_surname,
        &f.father_name,
        &f.mother_name,
        &f.spouse_name,
    ] {
        if let Some(ref s) = val {
            if !s.is_empty() {
                parts.push(s.clone());
            }
        }
    }
    if let Some(ref sc_json) = f.surname_candidates_json {
        if !sc_json.is_empty() && sc_json != "[]" {
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(sc_json) {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        if !s.is_empty() {
                            parts.push(s.to_string());
                        }
                    } else if let Some(obj) = item.as_object() {
                        for k in [
                            "raw_ocr",
                            "normalized_ocr",
                            "consensus_stem",
                            "resolved_surname",
                        ] {
                            if let Some(v) = obj.get(k).and_then(|v| v.as_str()) {
                                if !v.is_empty() {
                                    parts.push(v.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    parts
}

#[pyfunction]
#[pyo3(signature = (db_path, candidate_fact_ids=None, surname_query=None, q_query=None, fts_person_ids=vec![], fuzzy=true))]
pub fn score_person_facts_search_rs(
    py: Python<'_>,
    db_path: String,
    candidate_fact_ids: Option<Vec<i64>>,
    surname_query: Option<String>,
    q_query: Option<String>,
    fts_person_ids: Vec<i64>,
    fuzzy: bool,
) -> PyResult<(usize, Vec<(i64, f64, Option<i64>)>)> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let base_select = "SELECT
                pf.id, pf.census_sheet_id,
                pf.given_name_raw, pf.given_name_sk, pf.given_name_hu,
                pf.surname_raw, pf.surname_normalized, pf.mother_maiden_surname,
                pf.father_name, pf.mother_name, pf.spouse_name, pf.surname_candidates_json,
                pf.village_raw, pf.village_sk, pf.village_hu, pf.fs_place_standardized,
                pf.birth_place_raw, pf.birth_place_sk, pf.birth_place_municipality,
                pf.home_jurisdiction, pf.social_status_or_occupation, pf.occupation_position,
                pf.religion, pf.observationes_raw, pf.house_number,
                pf.event_year, pf.birth_year_est,
                rr.raw_ocr_combined, rr.raw_ocr_left, rr.raw_ocr_right
            FROM personfact pf
            LEFT JOIN recordrow rr ON rr.id = pf.row_id";

        let read_row = |r: &rusqlite::Row<'_>| -> Result<SearchableFactRow, rusqlite::Error> {
            let ocr_c: Option<String> = r.get(27)?;
            let ocr_l: Option<String> = r.get(28)?;
            let ocr_r: Option<String> = r.get(29)?;
            let mut ocr_parts: Vec<&str> = Vec::with_capacity(3);
            if let Some(ref s) = ocr_c {
                if !s.is_empty() {
                    ocr_parts.push(s);
                }
            }
            if let Some(ref s) = ocr_l {
                if !s.is_empty() {
                    ocr_parts.push(s);
                }
            }
            if let Some(ref s) = ocr_r {
                if !s.is_empty() {
                    ocr_parts.push(s);
                }
            }
            let ocr_txt = if ocr_parts.is_empty() {
                None
            } else {
                Some(ocr_parts.join(" "))
            };
            Ok(SearchableFactRow {
                id: r.get(0)?,
                census_sheet_id: r.get(1)?,
                given_name_raw: r.get(2)?,
                given_name_sk: r.get(3)?,
                given_name_hu: r.get(4)?,
                surname_raw: r.get(5)?,
                surname_normalized: r.get(6)?,
                mother_maiden_surname: r.get(7)?,
                father_name: r.get(8)?,
                mother_name: r.get(9)?,
                spouse_name: r.get(10)?,
                surname_candidates_json: r.get(11)?,
                village_raw: r.get(12)?,
                village_sk: r.get(13)?,
                village_hu: r.get(14)?,
                fs_place_standardized: r.get(15)?,
                birth_place_raw: r.get(16)?,
                birth_place_sk: r.get(17)?,
                birth_place_municipality: r.get(18)?,
                home_jurisdiction: r.get(19)?,
                social_status_or_occupation: r.get(20)?,
                occupation_position: r.get(21)?,
                religion: r.get(22)?,
                observationes_raw: r.get(23)?,
                house_number: r.get(24)?,
                event_year: r.get(25)?,
                birth_year_est: r.get(26)?,
                ocr_txt,
            })
        };

        let all_candidates: Vec<SearchableFactRow> = match candidate_fact_ids {
            Some(ref ids) if ids.is_empty() => Vec::new(),
            Some(ref ids) => {
                let mut rows_out: Vec<SearchableFactRow> = Vec::with_capacity(ids.len());
                for chunk in ids.chunks(500) {
                    let placeholders: Vec<String> =
                        (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                    let sql = format!(
                        "{} WHERE pf.id IN ({}) ORDER BY pf.id DESC",
                        base_select,
                        placeholders.join(", ")
                    );
                    let mut stmt = conn
                        .prepare(&sql)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let chunk_rows = stmt
                        .query_map(params_from_iter(chunk.iter()), read_row)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    rows_out.extend(chunk_rows);
                }
                rows_out.sort_by(|a, b| b.id.cmp(&a.id));
                rows_out
            }
            None => {
                let sql = format!("{} ORDER BY pf.id DESC", base_select);
                let mut stmt = conn
                    .prepare(&sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let rows_out = stmt
                    .query_map([], read_row)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                rows_out
            }
        };

        let total_candidates_count = all_candidates.len();
        let s_query_clean = surname_query
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let q_clean = q_query
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let fts_id_set: HashSet<i64> = fts_person_ids.into_iter().collect();
        let threshold = if fuzzy { 0.76 } else { 0.90 };
        let needle = q_clean
            .as_ref()
            .map(|q| fold_text_diacritics_internal(q))
            .unwrap_or_default();
        let q_toks: Vec<String> = q_clean
            .as_ref()
            .map(|q| {
                extract_name_tokens(q)
                    .into_iter()
                    .map(|t| fold_text_diacritics_internal(&t))
                    .filter(|t| !t.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let scored: Vec<(i64, f64, Option<i64>)> = all_candidates
            .par_iter()
            .filter_map(|f| {
                let struct_cands = extract_struct_name_cands(f);
                let mut current_score = 1.0_f64;

                if let Some(ref sq) = s_query_clean {
                    if fuzzy {
                        let mut s_cands: Vec<Option<String>> =
                            Vec::with_capacity(3 + struct_cands.len());
                        s_cands.push(f.surname_raw.clone());
                        s_cands.push(f.surname_normalized.clone());
                        s_cands.push(f.mother_maiden_surname.clone());
                        for sc in &struct_cands {
                            s_cands.push(Some(sc.clone()));
                        }
                        let sc = fuzzy_name_match_score_internal(sq, &s_cands, 0.76);
                        if sc < 0.76 {
                            return None;
                        }
                        current_score = sc;
                    }
                }

                if let Some(ref qq) = q_clean {
                    let mut haystack_parts: Vec<String> =
                        Vec::with_capacity(struct_cands.len() + 16);
                    haystack_parts.extend(struct_cands.iter().cloned());
                    if let Some(ref ot) = f.ocr_txt {
                        haystack_parts.push(ot.clone());
                    }
                    for opt_s in [
                        &f.village_raw,
                        &f.village_sk,
                        &f.village_hu,
                        &f.fs_place_standardized,
                        &f.birth_place_raw,
                        &f.birth_place_sk,
                        &f.birth_place_municipality,
                        &f.home_jurisdiction,
                        &f.social_status_or_occupation,
                        &f.occupation_position,
                        &f.religion,
                        &f.observationes_raw,
                        &f.house_number,
                    ] {
                        if let Some(ref s) = opt_s {
                            if !s.is_empty() {
                                haystack_parts.push(s.clone());
                            }
                        }
                    }
                    if let Some(ey) = f.event_year {
                        haystack_parts.push(ey.to_string());
                    }
                    if let Some(by) = f.birth_year_est {
                        haystack_parts.push(by.to_string());
                    }
                    let haystack = fold_text_diacritics_internal(&haystack_parts.join(" "));

                    let mut q_score = 0.0_f64;
                    if !needle.is_empty() && haystack.contains(&needle) {
                        q_score = 1.0;
                    } else {
                        let mut f_cands: Vec<Option<String>> =
                            Vec::with_capacity(struct_cands.len() + 7);
                        for sc in &struct_cands {
                            f_cands.push(Some(sc.clone()));
                        }
                        f_cands.push(f.village_raw.clone());
                        f_cands.push(f.village_sk.clone());
                        f_cands.push(f.birth_place_raw.clone());
                        f_cands.push(f.birth_place_sk.clone());
                        f_cands.push(f.birth_place_municipality.clone());
                        f_cands.push(f.social_status_or_occupation.clone());
                        f_cands.push(f.house_number.clone());

                        let f_sc = fuzzy_name_match_score_internal(qq, &f_cands, threshold);
                        if f_sc >= threshold {
                            q_score = f_sc;
                        } else if fts_id_set.contains(&f.id) {
                            let words: Vec<&str> = haystack.split_whitespace().collect();
                            if !q_toks.is_empty()
                                && q_toks
                                    .iter()
                                    .all(|qt| words.iter().any(|w| w.contains(qt.as_str())))
                            {
                                q_score = 0.88;
                            } else if let Some(ref ot) = f.ocr_txt {
                                let ocr_sc = fuzzy_name_match_score_internal(
                                    qq,
                                    &[Some(ot.clone())],
                                    threshold,
                                );
                                if ocr_sc >= threshold {
                                    q_score = ocr_sc;
                                }
                            }
                        } else if struct_cands.is_empty() {
                            if let Some(ref ot) = f.ocr_txt {
                                let ocr_sc = fuzzy_name_match_score_internal(
                                    qq,
                                    &[Some(ot.clone())],
                                    threshold,
                                );
                                if ocr_sc >= threshold {
                                    q_score = ocr_sc;
                                }
                            }
                        }
                    }

                    if q_score < threshold {
                        return None;
                    }
                    current_score = q_score;
                }

                Some((f.id, current_score, f.census_sheet_id))
            })
            .collect();

        let mut sorted = scored;
        if q_clean.is_some() {
            sorted.sort_by(|a, b| {
                let sa = (a.1 * 1000.0).round() as i64;
                let sb = (b.1 * 1000.0).round() as i64;
                (sb, b.0).cmp(&(sa, a.0))
            });
        } else {
            sorted.sort_by(|a, b| b.0.cmp(&a.0));
        }

        Ok((total_candidates_count, sorted))
    })
}

#[pyfunction]
#[pyo3(signature = (query, fuzzy=true))]
pub fn build_fts_match_expression_rs(query: &str, fuzzy: bool) -> String {
    let mut raw_tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in query.chars() {
        if is_name_token_char(ch) || ch == '-' || ch == '/' {
            cur.push(ch);
        } else if !cur.is_empty() {
            raw_tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        raw_tokens.push(cur);
    }

    let mut clauses: Vec<String> = Vec::new();
    for tok in raw_tokens {
        let clean: String = tok.chars().filter(|&c| is_name_token_char(c)).collect();
        if clean.is_empty() {
            continue;
        }
        let folded = clean_ascii_alnum(&clean);
        if folded.is_empty() {
            continue;
        }
        let mut variants: HashSet<String> = HashSet::new();
        variants.insert(folded.clone());
        variants.insert(format!("{}*", folded));

        let char_cnt = clean.chars().count();
        if clean.chars().all(|c| c.is_alphabetic()) && char_cnt >= 3 {
            let stem = surname_core_stem_internal(&clean);
            if !stem.is_empty() {
                variants.insert(stem.clone());
                variants.insert(format!("{}*", stem));
            }
            let base_tok = if !stem.is_empty() {
                stem.as_str()
            } else {
                clean.as_str()
            };
            if fuzzy {
                let dg = clean_ascii_alnum(&apply_digraphs_internal(base_tok));
                let orth = orthographic_fold_internal(base_tok);
                let skel = clean_ascii_alnum(&phonetic_skeleton_internal(base_tok));
                if dg.len() >= 3 {
                    variants.insert(dg.clone());
                    variants.insert(format!("{}*", dg));
                }
                if orth.len() >= 3 {
                    variants.insert(orth.clone());
                    variants.insert(format!("{}*", orth));
                }
                if skel.len() >= 3 {
                    variants.insert(skel);
                }
            }
            if let Some(cid) = resolve_given_cid_internal(&clean) {
                if let Some(eqs) = GIVEN_GROUPS.get(cid.as_str()) {
                    for &eq in eqs {
                        let eq_f = clean_ascii_alnum(eq);
                        if eq_f.len() >= 2 {
                            variants.insert(eq_f);
                        }
                    }
                }
            }
        }

        let mut safe_vec: Vec<String> = variants
            .into_iter()
            .filter(|v| v.bytes().any(|b| b.is_ascii_alphanumeric()))
            .collect();
        safe_vec.sort();
        if !safe_vec.is_empty() {
            clauses.push(format!("({})", safe_vec.join(" OR ")));
        }
    }
    clauses.join(" AND ")
}

#[pyfunction]
#[pyo3(signature = (db_path, q_query, year_from=None, year_to=None, municipality=None, district=None))]
pub fn match_catalog_sheets_and_overview_owners_rs(
    py: Python<'_>,
    db_path: String,
    q_query: String,
    year_from: Option<i64>,
    year_to: Option<i64>,
    municipality: Option<String>,
    district: Option<String>,
) -> PyResult<Vec<i64>> {
    py.allow_threads(move || {
        let q_clean = q_query.trim().to_string();
        if q_clean.is_empty() {
            return Ok(Vec::new());
        }
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let q_fold = fold_text_diacritics_internal(&q_clean);
        let m_fold_opt = municipality
            .as_deref()
            .map(|s| fold_text_diacritics_internal(s.trim()))
            .filter(|s| !s.is_empty());
        let d_fold_opt = district
            .as_deref()
            .map(|s| fold_text_diacritics_internal(s.trim()))
            .filter(|s| !s.is_empty());

        let mut ovh_sql = String::from(
            "SELECT census_sheet_id, matched_sheet_id, municipality,
                    owner_name, remarks, raw_ocr_text, house_number, census_year
             FROM censusoverviewhouse WHERE 1=1",
        );
        if let Some(yf) = year_from {
            ovh_sql.push_str(&format!(" AND census_year >= {}", yf));
        }
        if let Some(yt) = year_to {
            ovh_sql.push_str(&format!(" AND census_year <= {}", yt));
        }

        let mut ovh_rows: Vec<(
            Option<i64>,
            Option<i64>,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            i64,
        )> = Vec::new();
        {
            let mut stmt = conn
                .prepare(&ovh_sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get::<_, Option<i64>>(7)?.unwrap_or(1930),
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for r in rows.flatten() {
                if let Some(ref mf) = m_fold_opt {
                    if !fold_text_diacritics_internal(&r.2).contains(mf) {
                        continue;
                    }
                }
                ovh_rows.push(r);
            }
        }

        let ovh_matched_items: Vec<(Option<i64>, Option<i64>, String, Option<String>, i64)> = ovh_rows
            .par_iter()
            .filter_map(|(csid, msid, ov_muni, ov_owner, ov_rem, ov_ocr, ov_hn, ov_cyear)| {
                let mut ok = fuzzy_name_match_score_internal(
                    &q_clean,
                    &[ov_owner.clone(), ov_rem.clone()],
                    0.76,
                ) >= 0.76;
                if !ok {
                    if let Some(ref ocr_s) = ov_ocr {
                        if !q_fold.is_empty()
                            && fold_text_diacritics_internal(ocr_s).contains(&q_fold)
                        {
                            ok = true;
                        } else if ov_owner.as_deref().unwrap_or("").is_empty()
                            && fuzzy_name_match_score_internal(
                                &q_clean,
                                &[Some(ocr_s.clone())],
                                0.76,
                            ) >= 0.76
                        {
                            ok = true;
                        }
                    }
                }
                if ok {
                    Some((*csid, *msid, ov_muni.clone(), ov_hn.clone(), *ov_cyear))
                } else {
                    None
                }
            })
            .collect();

        let mut seen: HashSet<i64> = HashSet::new();
        let mut out_ids: Vec<i64> = Vec::new();
        let mut needed_muni_year_hn: HashSet<(String, i64, String)> = HashSet::new();
        for (csid_opt, msid_opt, ov_muni, ov_hn, ov_cyear) in ovh_matched_items {
            if let Some(csid) = csid_opt.filter(|&x| x > 0) {
                if seen.insert(csid) {
                    out_ids.push(csid);
                }
            }
            if let Some(msid) = msid_opt.filter(|&x| x > 0) {
                if seen.insert(msid) {
                    out_ids.push(msid);
                }
            }
            let m_canon = canonical_municipality_key_internal(Some(&ov_muni));
            let hn_norm = normalize_house_number_internal(ov_hn.as_deref());
            if !m_canon.is_empty() && !hn_norm.is_empty() {
                needed_muni_year_hn.insert((m_canon, ov_cyear, hn_norm));
            }
        }

        if !needed_muni_year_hn.is_empty() {
            let mut stmt = conn
                .prepare(
                    "SELECT id, municipality, census_year, house_number, sheet_type
                     FROM censussheet
                     WHERE house_number IS NOT NULL AND house_number != ''",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        r.get::<_, Option<i64>>(2)?.unwrap_or(1930),
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?
                            .unwrap_or_else(|| "household".to_string()),
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (sid, muni, cyear, hn, stype) in rows.flatten() {
                if matches!(stype.as_str(), "overview" | "municipal_overview") {
                    continue;
                }
                let m_canon = canonical_municipality_key_internal(Some(&muni));
                let hn_norm = normalize_house_number_internal(hn.as_deref());
                if needed_muni_year_hn.contains(&(m_canon, cyear, hn_norm)) && seen.insert(sid) {
                    out_ids.push(sid);
                }
            }
        }

        let mut cs_sql = String::from(
            "SELECT id, household_owner, municipality, municipality_folded,
                    district, district_folded
             FROM censussheet
             WHERE household_owner IS NOT NULL AND household_owner != ''",
        );
        if let Some(yf) = year_from {
            cs_sql.push_str(&format!(" AND census_year >= {}", yf));
        }
        if let Some(yt) = year_to {
            cs_sql.push_str(&format!(" AND census_year <= {}", yt));
        }

        let mut cs_rows: Vec<(i64, String)> = Vec::new();
        {
            let mut stmt = conn
                .prepare(&cs_sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (sid, owner, muni, muni_f, dist, dist_f) in rows.flatten() {
                if seen.contains(&sid) {
                    continue;
                }
                if let Some(ref mf) = m_fold_opt {
                    if !muni_f.contains(mf) && !fold_text_diacritics_internal(&muni).contains(mf) {
                        continue;
                    }
                }
                if let Some(ref df) = d_fold_opt {
                    if !dist_f.contains(df) && !fold_text_diacritics_internal(&dist).contains(df) {
                        continue;
                    }
                }
                cs_rows.push((sid, owner));
            }
        }

        let cs_matched: Vec<i64> = cs_rows
            .par_iter()
            .filter_map(|(sid, owner)| {
                if fuzzy_name_match_score_internal(&q_clean, &[Some(owner.clone())], 0.76) >= 0.76 {
                    Some(*sid)
                } else {
                    None
                }
            })
            .collect();

        for sid in cs_matched {
            if seen.insert(sid) {
                out_ids.push(sid);
            }
        }

        Ok(out_ids)
    })
}
