"""
Historical Matrika Lexicon & Observationes Parser for 18th-20th century Church Registers.
CRITICAL RULE: Personal names (given names and surnames) are NEVER translated or modernized
(e.g. Andreas stays Andreas, never Ondrej/András). OCR misreads MAY be repaired to the
most common FamilySearch spelling for the same area and period (Yargo -> Varga).
"""

from __future__ import annotations

import difflib
from functools import lru_cache
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable

# Known historical male given names (Latin, Hungarian, German/Slovak period forms + OCR variants)
KNOWN_MALE_GIVEN_NAMES: set[str] = {
    "andreas", "andrás", "stephanus", "stephannus", "stephans", "seghames", "istván",
    "joannes", "johannes", "joarnes", "joames", "jannes", "seanes", "svannes", "joanner", "jános",
    "josephus", "joseph", "józsef", "michael", "michaël", "mihály",
    "paulus", "saulus", "gaulus", "pál", "petrus", "péter",
    "franciscus", "francis", "ferenc", "georgius", "georgis", "györgy",
    "martinus", "márton", "mathias", "mátyás", "matthaeus", "matheng", "mathey", "máté",
    "nicolaus", "miklós", "antonius", "anton", "antal",
    "carolus", "coxolus", "károly", "ladislaus", "lászló", "thomas", "tamás",
    "albertus", "rudolphus", "andolphus", "andolph", "ludovicus", "lajos",
    "ignatius", "ignác", "emergicus", "emericus", "imre",
    "gabriel", "gábor", "samuel", "sámuel", "adamus", "ádám",
    "alexander", "sándor", "ferdinandus", "terdinandus",
    "sigismundus", "sigmanus", "vendelinus", "vendelius",
    "vincentius", "laurentius", "valentinus", "bernardus", "gregorius",
    # 20th-century Slovak / Czech census male given names
    "ján", "jan", "štefan", "stefan", "ondrej", "andrej", "jozef", "josef",
    "michal", "pavol", "pavel", "peter", "františek", "frantisek", "juraj",
    "martin", "matej", "matúš", "matus", "mikuláš", "mikulas", "karol", "karel",
    "ladislav", "tomáš", "tomas", "imrich", "vojtech", "ľudovít", "ludovit",
    "rudolf", "albert", "ferdinand", "žigmund", "zigmund", "vendelín", "vendelin",
    "vincent", "vavrinec", "valentín", "valentin", "bernard", "gregor", "adam",
    "milan", "jaroslav", "václav", "vaclav", "bohumil", "emil", "gejza",
    "koloman", "ernest", "július", "julius", "dezider", "viktor", "gustáv", "gustav",
    "alois", "alojz", "bartolomej", "cyril", "metod", "stanislav", "vladimír", "vladimir",
}

# Known historical female given names
KNOWN_FEMALE_GIVEN_NAMES: set[str] = {
    "maria", "mária", "anna", "catharina", "caharina", "catharin", "katalin",
    "elisabetha", "elizabetha", "elizabeth", "erzsébet",
    "susanna", "zsuzsanna", "veronica", "vexonica",
    "magdalena", "magdolna", "helena", "ilona", "eva", "éva",
    "barbara", "borbála", "margaritha", "margaretha", "margit",
    "theresia", "teréz", "agnes", "ágnes", "apollonia",
    "julianna", "johanna", "rosalia", "rozália", "sophia", "zsófia",
    "clara", "klára", "christina", "dorothea", "juditha", "ludovica",
    "hermina", "clementina", "francisca", "josepha", "antonia", "carolina", "paulina",
    # 20th-century Slovak / Czech census female given names
    "katarína", "katarina", "alžbeta", "alzbeta", "zuzana", "veronika",
    "magdaléna", "elena", "barbora", "margita", "terézia", "terezia",
    "agneša", "agnesa", "apolónia", "apolonia", "juliana", "júlia", "julia",
    "jana", "žofia", "zofia", "kristína", "kristina", "dorota", "judita",
    "ľudmila", "ludmila", "hermína", "františka", "frantiska", "jozefa", "jozefína",
    "antónia", "karolína", "paulína", "božena", "bozena", "emília", "emilia",
    "olga", "oľga", "irena", "marta", "gizela", "šarlota", "sarlota", "matilda",
    "valéria", "valeria", " Cecília", "cecília", "cecilia", "amália", "amalia",
    "vincencia", "vincentia",
}

# Combined set of known historical given names (used ONLY to identify which token is a given name
# in raw OCR and enforce semantic type safety; NEVER to modernize or translate the name).
KNOWN_HISTORICAL_GIVEN_NAMES: set[str] = KNOWN_MALE_GIVEN_NAMES | KNOWN_FEMALE_GIVEN_NAMES

# Obvious paleographic OCR character repairs within the same historical Latin / Slovak language
GIVEN_NAME_OCR_REPAIRS: dict[str, str] = {
    "joarnes": "Joannes",
    "joames": "Joannes",
    "jannes": "Joannes",
    "seanes": "Joannes",
    "svannes": "Joannes",
    "joanner": "Joannes",
    "stephannus": "Stephanus",
    "stephans": "Stephanus",
    "seghames": "Stephanus",
    "vexonica": "Veronica",
    "caharina": "Catharina",
    "catharin": "Catharina",
    "matheng": "Matthaeus",
    "coxolus": "Carolus",
    "andolphus": "Rudolphus",
    # 1930/1940 Slovak cursive census given-name OCR repairs
    "heter": "Štefan",
    "steena": "Irena",
    "stena": "Irena",
    "vivencia": "Vincencia",
    "havia": "Mária",
    "haqdalena": "Magdaléna",
    "doref": "Jozef",
    "joref": "Jozef",
}

# Historical month names (Latin, Hungarian, Slovak, German + abbreviations + common OCR misreads)
HISTORICAL_MONTH_MAP: dict[str, str] = {
    # January
    "januarius": "Januarius",
    "januarii": "Januarius",
    "januario": "Januarius",
    "ianuarius": "Januarius",
    "januar": "Januarius",
    "január": "Január",
    "jan": "Jan.",
    # February
    "februarius": "Februarius",
    "februarii": "Februarius",
    "februario": "Februarius",
    "februar": "Februarius",
    "február": "Február",
    "febr": "Febr.",
    "feb": "Feb.",
    # March
    "martius": "Martius",
    "martii": "Martius",
    "martio": "Martius",
    "március": "Március",
    "marec": "Marec",
    "mart": "Mart.",
    "mar": "Mar.",
    # April
    "aprilis": "Aprilis",
    "aprili": "Aprilis",
    "april": "April",
    "április": "Április",
    "apríl": "Apríl",
    "apr": "Apr.",
    # May
    "majus": "Majus",
    "maius": "Maius",
    "maji": "Majus",
    "maii": "Maius",
    "majo": "Majus",
    "május": "Május",
    "máj": "Máj",
    "maj": "Maj.",
    "mai": "Mai.",
    "may": "Majus",
    # June
    "junius": "Junius",
    "iunius": "Junius",
    "junii": "Junius",
    "iunii": "Junius",
    "junio": "Junius",
    "június": "Június",
    "jún": "Jún",
    "jun": "Jun.",
    "iun": "Jun.",
    # July (including common Kurrent OCR misread 'Sutius' / 'Surius' / 'Jutius')
    "julius": "Julius",
    "iulius": "Julius",
    "julii": "Julius",
    "iulii": "Julius",
    "julio": "Julius",
    "július": "Július",
    "júl": "Júl",
    "júla": "Júl",
    "júlá": "Júl",
    "jula": "Júl",
    "júli": "Júl",
    "juli": "Júl",
    "jul": "Jul.",
    "januára": "Január",
    "januara": "Január",
    "februára": "Február",
    "februara": "Február",
    "marca": "Marec",
    "apríla": "Apríl",
    "aprila": "Apríl",
    "mája": "Máj",
    "maja": "Máj",
    "júna": "Jún",
    "juna": "Jún",
    "júni": "Jún",
    "juni": "Jún",
    "augusta": "August",
    "septembra": "September",
    "októbra": "Október",
    "oktobra": "Október",
    "novembra": "November",
    "decembra": "December",
    "iul": "Jul.",
    "sutius": "Julius",
    "surius": "Julius",
    "jutius": "Julius",
    "jullius": "Julius",
    # August
    "augustus": "Augustus",
    "augusti": "Augustus",
    "augusto": "Augustus",
    "august": "August",
    "augusztus": "Augusztus",
    "auqustus": "Augustus",
    "aug": "Aug.",
    # September
    "september": "September",
    "septembris": "September",
    "septembri": "September",
    "szeptember": "Szeptember",
    "7ber": "7ber",
    "7bris": "7bris",
    "7br": "7br.",
    "sept": "Sept.",
    "sep": "Sep.",
    # October
    "october": "October",
    "octobris": "October",
    "octobri": "October",
    "október": "Október",
    "8ber": "8ber",
    "8bris": "8bris",
    "8br": "8br.",
    "oct": "Oct.",
    "okt": "Okt.",
    # November
    "november": "November",
    "novembris": "November",
    "novembri": "November",
    "9ber": "9ber",
    "9bris": "9bris",
    "9br": "9br.",
    "nov": "Nov.",
    # December
    "december": "December",
    "decembris": "December",
    "decembri": "December",
    "10ber": "10ber",
    "10bris": "10bris",
    "xber": "Xber",
    "xbris": "Xbris",
    "dec": "Dec.",
}

# Historical village OCR repair map (fixes OCR character misreads like Va'qlijhey -> Vágújhely
# for FamilySearch historical gazetteer lookup while keeping village_raw untouched).
VILLAGE_MAP: dict[str, tuple[str, str]] = {
    "vagujhely": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "vágújhely": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "va'gljhey": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "va'qlijhey": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "vaé níhhely": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "vae nihhely": ("Nové Mesto nad Váhom (Vágújhely)", "Vágújhely"),
    "kis rippény": ("Malé Ripňany (Kis-Répény)", "Kis-Répény"),
    "kisrippény": ("Malé Ripňany (Kis-Répény)", "Kis-Répény"),
    "kis rippeny": ("Malé Ripňany (Kis-Répény)", "Kis-Répény"),
    "kis dippeng": ("Malé Ripňany (Kis-Répény)", "Kis-Répény"),
    "nagy rippény": ("Veľké Ripňany", "Veľké Ripňany"),
    "nagyrippény": ("Veľké Ripňany", "Veľké Ripňany"),
    "neressensis": ("Norošovce (Neresznice)", "Neresznice"),
    "neresse": ("Norošovce (Neresznice)", "Neresznice"),
    "nezette": ("Norošovce (Neresznice)", "Neresznice"),
    "backolaj": ("Paczolaj", "Paczolaj"),
    "püspökfalu": ("Biskupová (Püspökfalu)", "Püspökfalu"),
    "podhrad": ("Podhradie (Podhrad)", "Podhrad"),
    "hradek": ("Hrádok (Hradek)", "Hradek"),
    "hra'dett": ("Hrádok (Hradek)", "Hradek"),
    "hradett": ("Hrádok (Hradek)", "Hradek"),
    "mosocz": ("Mošovce (Mosocz)", "Mosocz"),
    "moséer": ("Mošovce (Mosocz)", "Mosocz"),
    "kisellodré": ("Malé Modrové (Kis-Modró)", "Kis-Modró"),
    "hisellodro": ("Malé Modrové (Kis-Modró)", "Kis-Modró"),
    "radobica": ("Radobica", "Radobica"),
    "paczolaj": ("Paczolaj", "Paczolaj"),
    "sarluska": ("Šarluhy (Sarluska)", "Sarluska"),
    "sarkulka": ("Šarluhy (Sarluska)", "Sarluska"),
    "nahas": ("Naháč (Nahács)", "Nahács"),
    "nahács": ("Naháč (Nahács)", "Nahács"),
}

# Social status & occupations (Latin/Hungarian + common OCR variants -> readable explanation)
OCCUPATION_MAP: dict[str, str] = {
    "colonus": "colonus (sedliak / roľník)",
    "coloni": "coloni (sedliaci / roľníci)",
    "colonius": "colonus (sedliak / roľník)",
    "colm": "colonus (sedliak / roľník)",
    "colmi": "coloni (sedliaci / roľníci)",
    "columne": "coloni (sedliaci / roľníci)",
    "columnev": "coloni (sedliaci / roľníci)",
    "inquilinus": "inquilinus (želiar)",
    "inquilini": "inquilini (želiari)",
    "subinquilinus": "subinquilinus (podželiar)",
    "famulus": "famulus (sluha / čeľadník)",
    "famuli": "famuli (sluhovia / čeľadníci)",
    "familii": "famuli (sluhovia / čeľadníci)",
    "opilio": "opilio (pastier oviec)",
    "boarius": "boarius (pastier dobytka)",
    "boarins": "boarius (pastier dobytka)",
    "braius": "boarius (pastier dobytka)",
    "molitor": "molitor (mlynár)",
    "faber": "faber (kováč)",
    "faber ferrarius": "faber ferrarius (kováč)",
    "sutor": "sutor (obuvník / čižmár)",
    "sartor": "sartor (krajčír)",
    "cives": "cives (mešťan)",
    "civis": "civis (mešťan)",
    "miles": "miles (vojak)",
    "mercenarius": "mercenarius (nádenník)",
    "munarius": "mercenarius (nádenník)",
    "nobilis": "nobilis (šľachtic / zemian)",
    # 1930/1940 Slovak census occupations
    "roľník": "roľník",
    "rolnik": "roľník",
    "maloroľník": "maloroľník",
    "malorolnik": "maloroľník",
    "nádenník": "nádenník",
    "nadennik": "nádenník",
    "nádenníčka": "nádenníčka",
    "robotník": "robotník",
    "robotnik": "robotník",
    "poľnohosp. robotník": "poľnohospodársky robotník",
    "v domácnosti": "v domácnosti",
    "domácnosť": "v domácnosti",
    "domacnost": "v domácnosti",
    "murár": "murár",
    "murar": "murár",
    "tesár": "tesár",
    "tesar": "tesár",
    "obuvník": "obuvník",
    "obuvnik": "obuvník",
    "čižmár": "čižmár",
    "krajčír": "krajčír",
    "krajcir": "krajčír",
    "krajčírka": "krajčírka",
    "kováč": "kováč",
    "stolár": "stolár",
    "stolar": "stolár",
    "zámočník": "zámočník",
    "zamocnik": "zámočník",
    "železničiar": "železničiar",
    "zelezniciar": "železničiar",
    "obchodník": "obchodník",
    "obchodnik": "obchodník",
    "úradník": "úradník",
    "uradnik": "úradník",
    "učiteľ": "učiteľ",
    "ucitel": "učiteľ",
    "slúžka": "slúžka",
    "sluzka": "slúžka",
    "sluha": "sluha",
    "čeľadník": "čeľadník",
    "výmenkár": "výmenkár",
    "vymenkar": "výmenkár",
    "dôchodca": "dôchodca",
    "dochodca": "dôchodca",
    "žiak": "žiak",
    "študent": "študent",
}

# Raw tokens that represent occupations/titles/roles and must NEVER be treated as surnames or given names
OCCUPATION_OR_ROLE_TOKENS: set[str] = {
    "colonus", "coloni", "colonius", "colm", "colmi", "columne", "columnev", "cotton",
    "inquilinus", "inquilini", "subinquilinus", "famulus", "famuli", "familii",
    "opilio", "boarius", "boarins", "braius", "molitor", "faber", "ferrarius",
    "sutor", "sartor", "cives", "civis", "miles", "mercenarius", "munarius", "nobilis",
    "parochus", "parchus", "garochus", "capellanus", "capellaans", "portal",
    "uxor", "utox", "utor", "mor", "et", "ex", "coa", "tova",
    "roľník", "rolnik", "maloroľník", "malorolnik", "nádenník", "nadennik",
    "robotník", "robotnik", "domácnosti", "domácnosť", "domacnost", "murár", "tesár",
    "obuvník", "krajčír", "stolár", "zámočník", "železničiar", "obchodník", "úradník",
    "učiteľ", "slúžka", "sluha", "výmenkár", "dôchodca", "žiak", "študent",
    "prednosta", "manželka", "manzelka", "hlava", "majiteľ", "majitel",
    "syn", "dcéra", "dcera", "vdova", "vdovec", "slobodný", "slobodna", "ženatý", "vydatá",
}

RELIGION_MAP: dict[str, str] = {
    "r. cath.": "Rom. Cath. (rímskokatolícka)",
    "rom. cath.": "Rom. Cath. (rímskokatolícka)",
    "r.c.": "R. C. (rímskokatolícka)",
    "r. k.": "R. K. (rímskokatolícka)",
    "rím. kat.": "R. K. (rímskokatolícka)",
    "rim. kat.": "R. K. (rímskokatolícka)",
    "rímskokat": "R. K. (rímskokatolícka)",
    "rimskokat": "R. K. (rímskokatolícka)",
    "r cath": "Rom. Cath. (rímskokatolícka)",
    "k cath": "Rom. Cath. (rímskokatolícka)",
    "catholii": "Rom. Cath. (rímskokatolícka)",
    "rk": "R. K. (rímskokatolícka)",
    "ev. a.c.": "ev. a. c. (evanjelická a. v.)",
    "ev. a. v.": "ev. a. c. (evanjelická a. v.)",
    "e.a.c.": "e. a. c. (evanjelická a. v.)",
    "ev. a.e.": "ev. a. c. (evanjelická a. v.)",
    "cv. a.e.": "ev. a. c. (evanjelická a. v.)",
    "evang.": "evang. (evanjelická)",
    "evanj": "evang. (evanjelická)",
    "gr. cath.": "gr. cath. (gréckokatolícka)",
    "gr. kat.": "gr. cath. (gréckokatolícka)",
    "ref.": "ref. (reformovaná)",
    "isr.": "isr. (izraelitská)",
    "izrael": "isr. (izraelitská)",
    "žid.": "isr. (izraelitská)",
    "bez vyznania": "bez vyznania",
}

# Direct paleographic OCR repairs for surnames (Kurrent / period Hungarian-Slovak).
# Applied before similarity scoring so well-known misreads are deterministic.
SURNAME_OCR_REPAIRS: dict[str, str] = {
    "yargo": "Varga",
    "yarga": "Varga",
    "warga": "Varga",
    "vargo": "Varga",
    "farga": "Varga",
    "kovaasik": "Kovacsik",
    "kowacsik": "Kovacsik",
    "kowacs": "Kovács",
    "kovacs": "Kovács",
    "terenesik": "Terencsík",
    "trencsik": "Trencsík",
    "stasntna": "Šťastná",
    "stastnaova": "Šťastná",
    "stasntnaova": "Šťastná",
    "stastnyova": "Šťastná",
    "šťastnaová": "Šťastná",
    "šťastnáová": "Šťastná",
    "šťastnýová": "Šťastná",
    "siastna": "Šťastná",
    "siastnová": "Šťastná",
    "siastnova": "Šťastná",
    "hastua": "Šťastná",
    "hastuová": "Šťastná",
    "haštuová": "Šťastná",
    "řastná": "Šťastná",
    "rastna": "Šťastná",
    "srastna": "Šťastná",
    "stasnij": "Šťastný",
    "stasny": "Šťastný",
    "buchta-ová": "Buchtová",
    "buchta-ova": "Buchtová",
    "vančoová": "Vančová",
    "vancoova": "Vančová",
}

# Minimum weighted-similarity required to auto-correct an OCR token.
AUTO_CORRECT_THRESHOLD = 0.82

# Nyitra / Western Slovakia / Upper Hungary surnames attested on FamilySearch
# in 18th–19th century church records, with relative frequency weights.
_NYITRA_SURNAMES: dict[str, float] = {
    "Varga": 100, "Kovács": 95, "Kovacsik": 55,
    "Horváth": 90, "Tóth": 88, "Nagy": 86, "Molnár": 70, "Szabó": 80, "Balázs": 50,
    "Novotny": 60, "Mucska": 45, "Hutta": 40, "Hudec": 42, "Cserep": 35, "Terencsík": 30,
    "Trencsík": 28, "Havalda": 32, "Savlik": 30, "Schmitta": 25, "Jadák": 28,
    "Stravárik": 22, "Dávid": 35, "Lalus": 18, "Krajcsovics": 40, "Krajčovič": 20,
    "Németh": 55, "Kiss": 50, "Farkas": 48, "Papp": 40, "Takács": 45, "Mészáros": 38,
    "Juhász": 42, "Simon": 40, "Fekete": 36, "Lakatos": 30, "Oláh": 34, "Rácz": 32,
    "Barta": 30, "Balogh": 45, "Gál": 38, "Lukács": 28, "Kocsis": 26, "Boros": 24,
    "Hajdú": 22, "Szőke": 20, "Vincze": 28, "Orbán": 22, "Sörös": 16, "Cser": 18,
    "Hudák": 30, "Kováč": 40, "Kováčik": 28, "Horvát": 35, "Toth": 40, "Vargha": 25,
    "Selnák": 15, "Szelnák": 12, "Novotný": 35, "Novotna": 20, "Cserepka": 14,
    "Mucskai": 12, "Bala": 18, "Mikula": 22, "Kráľ": 65, "Bystrický": 40,
    "Bystrická": 40, "Šťastný": 55, "Šťastná": 55,
    "Paraba": 50, "Buchta": 45, "Liška": 45, "Škoda": 45, "Homola": 42,
    "Vacula": 42, "Vančo": 45, "Šimko": 45, "Bartko": 40, "Marko": 40,
    "Majerský": 42, "Majerská": 42,
}

_DEFAULT_SURNAMES: dict[str, float] = {
    **_NYITRA_SURNAMES,
    "Kovács": 100, "Nagy": 95, "Tóth": 92, "Szabó": 90, "Horváth": 88, "Varga": 85,
    "Kiss": 70, "Molnár": 68, "Németh": 60, "Farkas": 55, "Balogh": 50, "Papp": 45,
}

_TRENCSEN_SURNAMES: dict[str, float] = {
    **{k: v * 0.7 for k, v in _NYITRA_SURNAMES.items()},
    "Križan": 30, "Holub": 22, "Černý": 20, "Novák": 40, "Svoboda": 18,
}

_POZSONY_SURNAMES: dict[str, float] = {
    **{k: v * 0.7 for k, v in _NYITRA_SURNAMES.items()},
    "Schwartz": 25, "Weiss": 22, "Klein": 20, "Steiner": 18, "Bauer": 16,
}

# Period buckets: earlier registers lean more Latin/Hungarian spellings.
REGIONAL_FS_SURNAMES: dict[str, dict[str, dict[str, float]]] = {
    "nyitra": {
        "1700_1850": {k: v * 0.9 for k, v in _NYITRA_SURNAMES.items()},
        "1850_1920": dict(_NYITRA_SURNAMES),
    },
    "trencsen": {
        "1700_1850": {k: v * 0.9 for k, v in _TRENCSEN_SURNAMES.items()},
        "1850_1920": dict(_TRENCSEN_SURNAMES),
    },
    "pozsony": {
        "1700_1850": {k: v * 0.9 for k, v in _POZSONY_SURNAMES.items()},
        "1850_1920": dict(_POZSONY_SURNAMES),
    },
    "default": {
        "1700_1850": {k: v * 0.9 for k, v in _DEFAULT_SURNAMES.items()},
        "1850_1920": dict(_DEFAULT_SURNAMES),
    },
}

_NYITRA_GIVEN: dict[str, float] = {
    "Joannes": 100, "Maria": 98, "Anna": 90, "Stephanus": 80, "Josephus": 78,
    "Catharina": 75, "Andreas": 70, "Elisabetha": 65, "Paulus": 60, "Barbara": 50,
    "Michael": 55, "Franciscus": 52, "Georgius": 48, "Martinus": 45, "Helena": 40,
    "Eva": 38, "Petrus": 42, "Theresia": 36, "Mathias": 40, "Antonius": 38,
    "Carolus": 30, "Ladislaus": 28, "Thomas": 32, "Susanna": 35, "Veronica": 30,
    "Magdalena": 28, "Rosalia": 22, "Julianna": 24, "Agnes": 20, "Apollonia": 16,
    "Nicolaus": 26, "Ignatius": 18, "Ludovicus": 20, "Emericus": 16, "Gabriel": 18,
    "Samuel": 14, "Adamus": 12, "Alexander": 14, "Ferdinandus": 12, "Sigismundus": 10,
    "Matthaeus": 22, "Johanna": 18, "Clara": 14, "Sophia": 12, "Hermina": 10,
    "Clementina": 8, "Francisca": 16, "Josepha": 14, "Antonia": 12, "Carolina": 12,
    "Paulina": 10, "Rudolphus": 10, "Albertus": 8, "Vincentius": 10, "Laurentius": 10,
    "Valentinus": 12, "Bernardus": 8, "Gregorius": 8, "Vendelinus": 8,
}

REGIONAL_FS_GIVEN_NAMES: dict[str, dict[str, dict[str, float]]] = {
    "nyitra": {
        "1700_1850": dict(_NYITRA_GIVEN),
        "1850_1920": dict(_NYITRA_GIVEN),
    },
    "trencsen": {
        "1700_1850": dict(_NYITRA_GIVEN),
        "1850_1920": dict(_NYITRA_GIVEN),
    },
    "pozsony": {
        "1700_1850": dict(_NYITRA_GIVEN),
        "1850_1920": dict(_NYITRA_GIVEN),
    },
    "default": {
        "1700_1850": dict(_NYITRA_GIVEN),
        "1850_1920": dict(_NYITRA_GIVEN),
    },
}

_NYITRA_PLACE_MARKERS: tuple[str, ...] = (
    "nyitra", "nitra", "kis-répény",
    "kisrepeny", "rippény", "rippen", "ripňany", "ripnany", "vágújhely", "vagujhely",
    "va'gljhey", "va'qlijhey", "nove mesto", "nové mesto", "neresznice", "norošovce",
    "norosovce", "paczolaj", "püspökfalu", "puspokfalu", "biskupová", "biskupova",
    "podhrad", "podhradie", "hradek", "hrádok", "hradok", "mosocz", "mošovce",
    "mosovce", "modró", "modrove", "modrové", "radobica", "sarluska", "šarluhy",
    "sarluhy", "nahács", "naháč", "nahac", "topoľčany", "topolcany", "hisellodro",
    "kisellodré", "nezette", "neresse", "neressensis",
)

_TRENCSEN_PLACE_MARKERS: tuple[str, ...] = (
    "trencsén", "trencsen", "trenčín", "trencin", "vágtarnóc", "tarnóc",
)

_POZSONY_PLACE_MARKERS: tuple[str, ...] = (
    "pozsony", "pozson", "pressburg", "bratislava", "prešporok", "presporok",
)

# Symmetric Kurrent / cursive substitution costs (0 = identical, 1 = unrelated).
_CONFUSION_COST: dict[frozenset[str], float] = {
    frozenset({"y", "v"}): 0.12,
    frozenset({"y", "s"}): 0.12,
    frozenset({"y", "j"}): 0.15,
    frozenset({"y", "i"}): 0.15,
    frozenset({"y", "g"}): 0.20,
    frozenset({"y", "q"}): 0.22,
    frozenset({"j", "q"}): 0.20,
    frozenset({"o", "a"}): 0.15,
    frozenset({"u", "n"}): 0.15,
    frozenset({"u", "v"}): 0.15,
    frozenset({"u", "r"}): 0.18,
    frozenset({"n", "v"}): 0.18,
    frozenset({"m", "n"}): 0.20,
    frozenset({"m", "u"}): 0.25,
    frozenset({"m", "w"}): 0.25,
    frozenset({"m", "h"}): 0.18,
    frozenset({"c", "e"}): 0.18,
    frozenset({"c", "o"}): 0.20,
    frozenset({"e", "o"}): 0.18,
    frozenset({"c", "t"}): 0.28,
    frozenset({"c", "k"}): 0.18,
    frozenset({"k", "g"}): 0.28,
    frozenset({"r", "n"}): 0.18,
    frozenset({"l", "t"}): 0.16,
    frozenset({"l", "i"}): 0.15,
    frozenset({"l", "f"}): 0.20,
    frozenset({"l", "j"}): 0.20,
    frozenset({"i", "t"}): 0.16,
    frozenset({"i", "f"}): 0.20,
    frozenset({"i", "j"}): 0.15,
    frozenset({"t", "j"}): 0.20,
    frozenset({"f", "j"}): 0.22,
    frozenset({"g", "q"}): 0.18,
    frozenset({"g", "j"}): 0.20,
    frozenset({"v", "r"}): 0.18,
    frozenset({"v", "w"}): 0.08,
    frozenset({"v", "f"}): 0.22,
    frozenset({"s", "p"}): 0.25,
    frozenset({"s", "g"}): 0.28,
    frozenset({"s", "j"}): 0.22,
    frozenset({"s", "f"}): 0.25,
    frozenset({"s", "h"}): 0.22,
    frozenset({"t", "f"}): 0.25,
    frozenset({"b", "p"}): 0.20,
    frozenset({"b", "k"}): 0.24,
    frozenset({"h", "k"}): 0.30,
    frozenset({"h", "b"}): 0.28,
    frozenset({"d", "o"}): 0.30,
    frozenset({"z", "s"}): 0.22,
    frozenset({"z", "c"}): 0.28,
    frozenset({"z", "r"}): 0.20,
}

# Extended OCR confusion costs used specifically by search-time fuzzy matching
# (`weighted_edit_similarity` / `fuzzy_token_similarity`), without altering extraction-time auto-correction.
_SEARCH_CONFUSION_COST: dict[frozenset[str], float] = {
    **_CONFUSION_COST,
    frozenset({"d", "t"}): 0.18,
    frozenset({"z", "s"}): 0.14,
    frozenset({"z", "c"}): 0.20,
    frozenset({"h", "k"}): 0.20,
    frozenset({"h", "b"}): 0.20,
    frozenset({"t", "f"}): 0.18,
}

_DIACRITIC_FOLD = str.maketrans(
    "áéíóôöőúüűýÁÉÍÓÔÖŐÚÜŰÝäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽ",
    "aeioooouuuyAEIOOOOUUUYaeAEcdllnrrstzCDLLNRRSTZ",
)

_FS_NAME_CACHE: dict[str, Any] | None = None
_FS_NAME_CACHE_PATH: Path | None = None
_FS_NAME_CACHE_LOCK = threading.Lock()


def _fs_cache_path() -> Path:
    if _FS_NAME_CACHE_PATH is not None:
        return _FS_NAME_CACHE_PATH
    from backend.app.config import settings

    settings.ensure_dirs()
    return settings.data_dir / "fs_area_names_cache.json"


def reset_fs_name_cache_for_tests(path: Path | None = None) -> None:
    """Test helper: point the FamilySearch name cache at a temp file (or empty memory)."""
    global _FS_NAME_CACHE, _FS_NAME_CACHE_PATH
    with _FS_NAME_CACHE_LOCK:
        _FS_NAME_CACHE_PATH = path
        _FS_NAME_CACHE = {"areas": {}} if path is None else None
        _MATCH_SURNAME_MEMO.clear()
        _MATCH_GIVEN_MEMO.clear()
        _CANDIDATE_WEIGHTS_MEMO.clear()


def _load_fs_name_cache() -> dict[str, Any]:
    global _FS_NAME_CACHE
    with _FS_NAME_CACHE_LOCK:
        if _FS_NAME_CACHE is not None:
            return _FS_NAME_CACHE
        path = _fs_cache_path()
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    _FS_NAME_CACHE = loaded
                    if "areas" not in _FS_NAME_CACHE:
                        _FS_NAME_CACHE["areas"] = {}
                    return _FS_NAME_CACHE
            except (OSError, json.JSONDecodeError):
                pass
        _FS_NAME_CACHE = {"areas": {}}
        return _FS_NAME_CACHE


def _save_fs_name_cache(cache: dict[str, Any]) -> None:
    path = _fs_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def resolve_area_key(place: str | None) -> str:
    """Map a village / FS place string onto a regional lexicon key."""
    if not place:
        return "nyitra"
    low = place.lower()
    if any(m in low for m in _TRENCSEN_PLACE_MARKERS):
        return "trencsen"
    if any(m in low for m in _POZSONY_PLACE_MARKERS):
        return "pozsony"
    if any(m in low for m in _NYITRA_PLACE_MARKERS):
        return "nyitra"
    return "nyitra"


def resolve_period_key(year: int | str | None) -> str:
    try:
        y = int(str(year).strip())
    except (TypeError, ValueError):
        return "1850_1920"
    if y < 1850:
        return "1700_1850"
    return "1850_1920"


def _cache_bucket_key(area: str, period: str) -> str:
    return f"{area}|{period}"


def fs_name_cache_size(area: str | None, year: int | str | None) -> int:
    area_key = resolve_area_key(area)
    period = resolve_period_key(year)
    cache = _load_fs_name_cache()
    bucket = (cache.get("areas") or {}).get(_cache_bucket_key(area_key, period)) or {}
    surnames = bucket.get("surnames") or {}
    given = bucket.get("given_names") or {}
    return int(len(surnames) + len(given))


def record_observed_fs_names(
    area: str | None,
    year: int | str | None,
    surnames: list[str] | None = None,
    given_names: list[str] | None = None,
) -> None:
    """Merge observed FamilySearch hit names into the persistent regional cache."""
    area_key = resolve_area_key(area)
    period = resolve_period_key(year)
    cache = _load_fs_name_cache()
    areas = cache.setdefault("areas", {})
    bucket = areas.setdefault(_cache_bucket_key(area_key, period), {"surnames": {}, "given_names": {}})
    for raw in surnames or []:
        cleaned = _clean_name_token(raw)
        if not cleaned or len(cleaned) < 3:
            continue
        if is_month_name(cleaned) or is_occupation_or_role_token(cleaned):
            continue
        canon = cleaned[0].upper() + cleaned[1:]
        bucket["surnames"][canon] = int(bucket["surnames"].get(canon, 0)) + 1
    for raw in given_names or []:
        cleaned = _clean_name_token(raw)
        if not cleaned or len(cleaned) < 3:
            continue
        if is_month_name(cleaned) or is_occupation_or_role_token(cleaned):
            continue
        canon = cleaned[0].upper() + cleaned[1:]
        bucket["given_names"][canon] = int(bucket["given_names"].get(canon, 0)) + 1
    with _FS_NAME_CACHE_LOCK:
        global _FS_NAME_CACHE
        _FS_NAME_CACHE = cache
        _MATCH_SURNAME_MEMO.clear()
        _MATCH_GIVEN_MEMO.clear()
        _CANDIDATE_WEIGHTS_MEMO.clear()
        _save_fs_name_cache(cache)


def _clean_name_token(token: str | None) -> str | None:
    if not token:
        return None
    cleaned = re.sub(
        r"[^a-zA-ZáéíóôöőúüűýÁÉÍÓÔÖŐÚÜŰÝäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽ\-]",
        "",
        token,
    ).strip(" .-")
    return cleaned or None


@lru_cache(maxsize=65536)
def _apply_digraphs(token: str) -> str:
    s = token.translate(_DIACRITIC_FOLD).lower()
    for src, dst in (
        ("cz", "cs"),
        ("tz", "cs"),
        ("sz", "s"),
        ("zs", "z"),
        ("dy", "gy"),
        ("nj", "ny"),
        ("lj", "ly"),
        ("th", "t"),
    ):
        s = s.replace(src, dst)
    s = s.replace("w", "v")
    return s


@lru_cache(maxsize=65536)
def _phonetic_skeleton(token: str) -> str:
    """Consonant skeleton used to block obviously unrelated candidates."""
    s = _apply_digraphs(token)
    s = re.sub(r"[aeiouy]+", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _CONFUSION_COST.get(frozenset({a, b}), 1.0)


@lru_cache(maxsize=65536)
def _paleographic_distance(left: str, right: str) -> float:
    """Weighted Levenshtein distance using Kurrent/cursive confusion costs."""
    a = _apply_digraphs(left)
    b = _apply_digraphs(right)
    n, m = len(a), len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    prev = [float(j) * 0.9 for j in range(m + 1)]
    for i, ca in enumerate(a, start=1):
        curr = [float(i) * 0.9]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 0.9,
                    curr[j - 1] + 0.9,
                    prev[j - 1] + _sub_cost(ca, cb),
                )
            )
        prev = curr
    return prev[-1]


@lru_cache(maxsize=65536)
def _paleographic_similarity(left: str, right: str) -> float:
    a = _apply_digraphs(left)
    b = _apply_digraphs(right)
    if not a and not b:
        return 1.0
    dist = _paleographic_distance(left, right)
    return max(0.0, 1.0 - dist / max(len(a), len(b), 1))


_CANDIDATE_WEIGHTS_MEMO: dict[tuple[str, str, str], dict[str, float]] = {}
_MATCH_SURNAME_MEMO: dict[tuple[str, str, str], tuple[str | None, float]] = {}
_MATCH_GIVEN_MEMO: dict[tuple[str, str, str], tuple[str | None, float]] = {}


def _merge_candidate_weights(
    regional: dict[str, dict[str, dict[str, float]]],
    area: str,
    period: str,
    cache_field: str,
) -> dict[str, float]:
    cache_key = (cache_field, area, period)
    cached_weights = _CANDIDATE_WEIGHTS_MEMO.get(cache_key)
    if cached_weights is not None:
        return cached_weights

    weights: dict[str, float] = {}

    def add(table: dict[str, float] | None, scale: float) -> None:
        if not table:
            return
        for name, w in table.items():
            weights[name] = max(weights.get(name, 0.0), float(w) * scale)

    for area_key in (area, "default"):
        region = regional.get(area_key) or {}
        add(region.get(period), 1.0)
        for p_key, table in region.items():
            if p_key != period:
                add(table, 0.55)
    cache = _load_fs_name_cache()
    bucket = (cache.get("areas") or {}).get(_cache_bucket_key(area, period)) or {}
    cached_table = bucket.get(cache_field) or {}
    add({k: float(v) + 8.0 for k, v in cached_table.items()}, 1.15)
    neighbor_period = "1700_1850" if period == "1850_1920" else "1850_1920"
    neighbor = (cache.get("areas") or {}).get(_cache_bucket_key(area, neighbor_period)) or {}
    add({k: float(v) + 4.0 for k, v in (neighbor.get(cache_field) or {}).items()}, 0.7)
    if cache_field == "given_names":
        for name in KNOWN_HISTORICAL_GIVEN_NAMES:
            canonical = GIVEN_NAME_OCR_REPAIRS.get(name, name[0].upper() + name[1:] if name else name)
            if canonical and canonical[0].islower():
                canonical = canonical[0].upper() + canonical[1:]
            weights.setdefault(canonical, 8.0)
    _CANDIDATE_WEIGHTS_MEMO[cache_key] = weights
    return weights


def _score_against_lexicon(
    raw: str,
    candidates: dict[str, float],
) -> tuple[str | None, float]:
    raw_clean = _clean_name_token(raw)
    if not raw_clean or len(raw_clean) < 3:
        return None, 0.0
    raw_low = raw_clean.lower()
    raw_folded = _apply_digraphs(raw_clean)
    raw_skel = _phonetic_skeleton(raw_clean)

    for name in candidates:
        if name.lower() == raw_low:
            return name, 1.0
    for name in candidates:
        if _apply_digraphs(name) == raw_folded:
            return name, 0.98

    max_w = max(candidates.values()) if candidates else 1.0
    best_name: str | None = None
    best_score = 0.0
    for name, weight in candidates.items():
        cand_skel = _phonetic_skeleton(name)
        if raw_skel and cand_skel:
            skel_ratio = min(len(raw_skel), len(cand_skel)) / max(len(raw_skel), len(cand_skel))
            if skel_ratio < 0.45 and _paleographic_similarity(raw_clean, name) < 0.9:
                continue
        sim = _paleographic_similarity(raw_clean, name)
        if sim < AUTO_CORRECT_THRESHOLD - 0.08:
            continue
        freq_norm = math.log(1.0 + float(weight)) / math.log(1.0 + max_w) if max_w > 0 else 0.0
        combined = 0.88 * sim + 0.12 * freq_norm
        if combined > best_score:
            best_score = combined
            best_name = name
    if best_name and _paleographic_similarity(raw_clean, best_name) >= AUTO_CORRECT_THRESHOLD:
        return best_name, max(best_score, _paleographic_similarity(raw_clean, best_name))
    return None, 0.0


def match_surname_to_area_period(
    raw_surname: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> tuple[str | None, float]:
    """Return the best regional/period FamilySearch surname for an OCR token."""
    raw_clean = _clean_name_token(raw_surname)
    if not raw_clean or len(raw_clean) < 3:
        return None, 0.0
    if is_month_name(raw_clean) or is_occupation_or_role_token(raw_clean):
        return None, 0.0
    low = raw_clean.lower()
    if low in SURNAME_OCR_REPAIRS:
        return SURNAME_OCR_REPAIRS[low], 1.0
    area_key = resolve_area_key(area)
    period = resolve_period_key(year)
    memo_key = (raw_clean, area_key, period)
    if memo_key in _MATCH_SURNAME_MEMO:
        return _MATCH_SURNAME_MEMO[memo_key]
    candidates = _merge_candidate_weights(REGIONAL_FS_SURNAMES, area_key, period, "surnames")
    res = _score_against_lexicon(raw_clean, candidates)
    if len(_MATCH_SURNAME_MEMO) < 4096:
        _MATCH_SURNAME_MEMO[memo_key] = res
    return res


def match_given_name_to_area_period(
    raw_given: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> tuple[str | None, float]:
    """Return the best regional/period historical given name for an OCR token.

    Never translates a known historical form (Andreas stays Andreas).
    """
    raw_clean = _clean_name_token(raw_given)
    if not raw_clean or len(raw_clean) < 3:
        return None, 0.0
    if is_month_name(raw_clean) or is_occupation_or_role_token(raw_clean):
        return None, 0.0
    low = raw_clean.lower()
    if low in GIVEN_NAME_OCR_REPAIRS:
        return GIVEN_NAME_OCR_REPAIRS[low], 1.0
    if low in KNOWN_HISTORICAL_GIVEN_NAMES:
        return raw_clean, 1.0
    area_key = resolve_area_key(area)
    period = resolve_period_key(year)
    memo_key = (raw_clean, area_key, period)
    if memo_key in _MATCH_GIVEN_MEMO:
        return _MATCH_GIVEN_MEMO[memo_key]
    candidates = _merge_candidate_weights(REGIONAL_FS_GIVEN_NAMES, area_key, period, "given_names")
    res = _score_against_lexicon(raw_clean, candidates)
    if len(_MATCH_GIVEN_MEMO) < 4096:
        _MATCH_GIVEN_MEMO[memo_key] = res
    return res


def name_match_repair_notes(original: str | None, corrected: str | None) -> list[str]:
    """Describe token-level FamilySearch area/period corrections for the audit trail."""
    if not original or not corrected or original == corrected:
        return []
    orig_toks = original.split()
    corr_toks = corrected.split()
    notes: list[str] = []
    for o, c in zip(orig_toks, corr_toks):
        if o.lower() == c.lower():
            continue
        kind = "given name" if is_known_given_name(c) else "surname"
        notes.append(f"Matched {kind} {o!r} -> {c!r} (FamilySearch area/period match)")
    if not notes and original.lower() != corrected.lower():
        notes.append(
            f"Matched name {original!r} -> {corrected!r} (FamilySearch area/period match)"
        )
    return notes


def is_month_name(token: str | None) -> bool:
    """Returns True if token is a Latin, Hungarian, Slovak, or German month name (or known OCR variant)."""
    if not token:
        return False
    clean = re.sub(r"[^a-zA-Z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]", "", token).lower()
    return clean in HISTORICAL_MONTH_MAP


def normalize_month_name(token: str | None) -> str | None:
    """Returns canonical historical month name if token is a known month or month OCR misread."""
    if not token:
        return None
    clean = re.sub(r"[^a-zA-Z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]", "", token).lower()
    return HISTORICAL_MONTH_MAP.get(clean)


def is_occupation_or_role_token(token: str | None) -> bool:
    """Returns True if token is an occupation, social status, or Latin relational keyword."""
    if not token:
        return False
    clean = re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]", "", token).lower()
    return clean in OCCUPATION_OR_ROLE_TOKENS


def infer_sex_from_given_name(given_name: str | None) -> str | None:
    """Infers 'M' or 'F' from a historical given name using the lexicon first, then Latin grammar."""
    if not given_name:
        return None
    tokens = [
        re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄË]", "", t).lower()
        for t in given_name.split()
    ]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    first = tokens[0]
    if first in KNOWN_MALE_GIVEN_NAMES:
        return "M"
    if first in KNOWN_FEMALE_GIVEN_NAMES:
        return "F"
    if first.endswith(("us", "as", "es", "el", "is", "o")):
        return "M"
    if first.endswith(("a", "ia", "na")):
        return "F"
    return None


def normalize_given_name(
    raw_name: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> tuple[str | None, str | None]:
    """
    Preserves the exact historical given name as written in the matrika without
    ever translating or changing old variants to modern variants.
    Only repairs obvious OCR character glitches (like Joarnes/Joames -> Joannes
    or area/period FamilySearch matches) and rejects month names, occupations,
    or pure numbers.
    """
    if not raw_name:
        return None, None
    # Strip leading entry/day numbers like '25° Josephus'
    stripped = re.sub(r"^\s*\d{1,3}\s*[°a-zA-Z]?\s*\.?\s+", "", raw_name)
    cleaned = re.sub(
        r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ\-\s.]",
        "",
        stripped,
    ).strip(" .,;-")
    if not cleaned:
        return None, None
    low = cleaned.lower()
    # Never accept a month name or occupation as a given name
    if is_month_name(low) or is_occupation_or_role_token(low):
        return None, None
    if low in GIVEN_NAME_OCR_REPAIRS:
        cleaned = GIVEN_NAME_OCR_REPAIRS[low]
        return cleaned, cleaned
    if low in KNOWN_HISTORICAL_GIVEN_NAMES:
        return cleaned, cleaned
    matched, score = match_given_name_to_area_period(cleaned, area=area, year=year)
    if matched and score >= AUTO_CORRECT_THRESHOLD:
        return matched, matched
    return cleaned, cleaned


def normalize_surname(
    raw_surname: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> tuple[str | None, str | None]:
    """Repairs OCR surname misreads using the regional FamilySearch lexicon.

    Does not translate or modernize a surname that already matches a known
    historical form; only substitutes paleographic misreads (Yargo -> Varga).
    """
    if not raw_surname:
        return None, None
    sanitized = sanitize_corrupted_surname_suffix(raw_surname)
    cleaned = _clean_name_token(sanitized or raw_surname)
    if not cleaned:
        return None, None
    if is_month_name(cleaned) or is_occupation_or_role_token(cleaned):
        return None, None
    matched, score = match_surname_to_area_period(cleaned, area=area, year=year)
    if matched and score >= AUTO_CORRECT_THRESHOLD:
        return matched, matched
    return cleaned, cleaned


def is_known_given_name(token: str) -> bool:
    if not token:
        return False
    low = re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄË]", "", token).lower()
    if not low or is_month_name(low) or is_occupation_or_role_token(low):
        return False
    if low in KNOWN_HISTORICAL_GIVEN_NAMES or low in GIVEN_NAME_OCR_REPAIRS:
        return True
    matched, score = match_given_name_to_area_period(token)
    return bool(matched and score >= AUTO_CORRECT_THRESHOLD)


def correct_names_in_text(
    text: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> str:
    """Replace surname/given tokens in a free-text cell with area/period matches."""
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        if is_occupation_or_role_token(tok) or is_month_name(tok):
            return tok
        given, _ = normalize_given_name(tok, area=area, year=year)
        if given and given.lower() != tok.lower() and is_known_given_name(given):
            return given
        if is_known_given_name(tok):
            return given or tok
        surname, _ = normalize_surname(tok, area=area, year=year)
        if surname and surname.lower() != tok.lower():
            return surname
        return tok

    return re.sub(
        r"[A-Za-záéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+",
        _repl,
        text,
    )


def apply_area_period_name_corrections(
    fields: dict[str, Any],
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> list[str]:
    """Auto-correct given names and surnames on an extraction dict; return repair notes."""
    area_hint = (
        area
        or fields.get("village_hu")
        or fields.get("village_sk")
        or fields.get("village_raw")
    )
    year_hint = year if year is not None else fields.get("event_year")
    repairs: list[str] = []

    for key in ("given_name_raw", "given_name_sk", "given_name_hu"):
        raw = fields.get(key)
        if not raw:
            continue
        corrected, _ = normalize_given_name(str(raw), area=area_hint, year=year_hint)
        if corrected and corrected != raw:
            repairs.extend(name_match_repair_notes(str(raw), corrected))
            fields[key] = corrected

    for key in ("surname_raw", "surname_normalized", "mother_maiden_surname"):
        raw = fields.get(key)
        if not raw:
            continue
        corrected, _ = normalize_surname(str(raw), area=area_hint, year=year_hint)
        if corrected and corrected != raw:
            repairs.extend(name_match_repair_notes(str(raw), corrected))
            fields[key] = corrected

    for key in ("father_name", "mother_name", "spouse_name"):
        raw = fields.get(key)
        if not raw:
            continue
        corrected = correct_names_in_text(str(raw), area=area_hint, year=year_hint)
        if corrected and corrected != raw:
            repairs.extend(name_match_repair_notes(str(raw), corrected))
            fields[key] = corrected

    if fields.get("surname_raw") and not fields.get("surname_normalized"):
        fields["surname_normalized"] = fields["surname_raw"]
    return repairs


def normalize_village(raw_village: str | None) -> tuple[str | None, str | None, str | None]:
    """Extracts house number and preserves/repairs historical village name without modernizing personal names."""
    if not raw_village:
        return None, None, None
    house_match = re.search(
        r"(?:Nro\.?|No\.?|N\.?|№|#|\bč\.?|\bH)\s*(\d{1,3}[a-zA-Z]?)|\b(\d{1,3})\s*\.?\s*$",
        raw_village,
        re.IGNORECASE,
    )
    house_no = None
    if house_match:
        cand_no = house_match.group(1) or house_match.group(2)
        if cand_no:
            num_digits = re.sub(r"\D", "", cand_no)
            # Avoid treating 4-digit years (e.g. 1884, 1749) as house numbers
            if num_digits and int(num_digits) <= 999:
                house_no = cand_no

    place_part = re.sub(
        r"(?:Nro\.?|No\.?|N\.?|№|#|\bH)\s*\d+[a-zA-Z]?.*",
        "",
        raw_village,
        flags=re.IGNORECASE,
    )
    place_part = re.sub(r"\b\d+\s*\.?\s*$", "", place_part).strip(" .,;-")
    key = place_part.lower().strip()
    if key in VILLAGE_MAP:
        v_orig, v_hu = VILLAGE_MAP[key]
        return v_orig, v_hu, house_no
    close = difflib.get_close_matches(key, list(VILLAGE_MAP.keys()), n=1, cutoff=0.6)
    if close:
        v_orig, v_hu = VILLAGE_MAP[close[0]]
        return v_orig, v_hu, house_no
    return (place_part or None), (place_part or None), house_no


_COMPILED_OCCUPATION_ITEMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in OCCUPATION_MAP.items()
]


def normalize_occupation(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    low = raw_text.lower()
    found: list[str] = []
    for pat, v in _COMPILED_OCCUPATION_ITEMS:
        if pat.search(low) and v not in found:
            found.append(v)
    return ", ".join(found) if found else None


def normalize_religion(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    low = raw_text.lower().strip()
    for k, v in RELIGION_MAP.items():
        if k in low:
            return v
    if re.search(
        r"(?:^|\b)(?:r|rím|rim|2|7)\.?\s*(?:k|h|d|n|u\.?|\?)\s*at\.?(?:\b|$)",
        low,
    ):
        return "R. K. (rímskokatolícka)"
    return None


def _extract_clause_year(text: str, keyword_pattern: str, year_pattern: str) -> int | None:
    # First look within the semicolon/pipe-separated clause that matched the keyword
    for clause in re.split(r"[;|\n]+", text):
        if re.search(keyword_pattern, clause, re.IGNORECASE):
            ym = re.search(year_pattern, clause)
            if ym:
                return int(ym.group(1))
    # Fallback: look after the keyword match in the full text, then anywhere in text
    km = re.search(keyword_pattern, text, re.IGNORECASE)
    if km:
        after = text[km.start() :]
        ym = re.search(year_pattern, after)
        if ym:
            return int(ym.group(1))
    ym_any = re.search(year_pattern, text)
    return int(ym_any.group(1)) if ym_any else None


def parse_observationes_events(obs_text: str | None) -> list[dict[str, Any]]:
    if not obs_text or len(obs_text.strip()) < 2:
        return []
    text = obs_text.strip()
    events: list[dict[str, Any]] = []

    death_pat = r"(\+|†|\bmortu|\bobiit|\bmeghalt|\bzomrel)"
    if re.search(death_pat, text, re.IGNORECASE):
        year_val = _extract_clause_year(text, death_pat, r"\b(1[789]\d{2}|20\d{2})\b")
        events.append({
            "event_type": "death",
            "type": "death",
            "label_sk": "Úmrtie zapísané v poznámke (Observationes)",
            "raw_note": text,
            "year": year_val,
            "date": str(year_val) if year_val else text,
        })

    marr_pat = r"(matrimon|contraxit|copulat|házasság|esküv|sobáš)"
    if re.search(marr_pat, text, re.IGNORECASE):
        year_val = _extract_clause_year(text, marr_pat, r"\b(1[89]\d{2}|20\d{2})\b")
        events.append({
            "event_type": "marriage",
            "type": "marriage",
            "label_sk": "Sobáš zapísaný v poznámke (Observationes)",
            "raw_note": text,
            "year": year_val,
            "date": str(year_val) if year_val else text,
        })

    return events


# Cross-language given name equivalence groups for search & local Matrika<->Census cross-matching.
# Note: Extraction NEVER translates raw historical names; these groups are used strictly for
# canonical search indexing and cross-matching across Latin, Hungarian, Slovak, and German forms.
GIVEN_NAME_EQUIVALENCE_GROUPS: dict[str, set[str]] = {
    "JOANNES": {"joannes", "johannes", "jános", "janos", "ján", "jan", "johann", "hans", "janko", "jano", "yoannes"},
    "STEPHANUS": {"stephanus", "istván", "istvan", "štefan", "stefan", "stephan", "pišta", "pista"},
    "ANDREAS": {"andreas", "andrás", "andras", "ondrej", "andrej", "bandi", "endrik"},
    "JOSEPHUS": {"josephus", "joseph", "józsef", "jozsef", "jozef", "josef", "jožo", "jozo"},
    "MICHAEL": {"michael", "michaël", "mihály", "mihaly", "michal", "mišo", "miso"},
    "PAULUS": {"paulus", "pál", "pal", "pavol", "pavel", "paul", "paľo", "palo"},
    "PETRUS": {"petrus", "péter", "peter", "peťo", "peto"},
    "FRANCISCUS": {"franciscus", "francis", "ferenc", "ferencz", "františek", "frantisek", "franz", "fero"},
    "GEORGIUS": {"georgius", "györgy", "gyorgy", "juraj", "ďuro", "duro", "georg", "jura"},
    "MARTINUS": {"martinus", "márton", "marton", "martin"},
    "MATHIAS": {"mathias", "mátyás", "matyas", "matthaeus", "máté", "matej", "matúš", "matus"},
    "NICOLAUS": {"nicolaus", "miklós", "miklos", "mikuláš", "mikulas", "nikolaus"},
    "ANTONIUS": {"antonius", "anton", "antal", "tonko"},
    "CAROLUS": {"carolus", "károly", "karoly", "karol", "karel", "karl"},
    "LADISLAUS": {"ladislaus", "lászló", "laszlo", "ladislav", "laco"},
    "THOMAS": {"thomas", "tamás", "tamas", "tomáš", "tomas"},
    "EMERICUS": {"emericus", "emergicus", "imre", "imrich"},
    "LUDOVICUS": {"ludovicus", "lajos", "ľudovít", "ludovit", "ludwig"},
    "IGNATIUS": {"ignatius", "ignác", "ignac", "ignaz"},
    "ADALBERTUS": {"adalbertus", "béla", "bela", "vojtech", "albertus", "albert"},
    "ALEXANDER": {"alexander", "sándor", "sandor", "šándor"},
    "SIGISMUNDUS": {"sigismundus", "zsigmond", "žigmund", "zigmund", "sigmund"},
    "JULIUS": {"julius", "gyula", "július"},
    "MARIA": {"maria", "mária", "mara", "marika"},
    "ANNA": {"anna", "anca", "anka", "annuška", "ana"},
    "CATHARINA": {"catharina", "katalin", "katarína", "katarina", "kata", "katka"},
    "ELISABETHA": {"elisabetha", "elizabetha", "elizabeth", "erzsébet", "erzsebet", "alžbeta", "alzbeta", "beta"},
    "SUSANNA": {"susanna", "zsuzsanna", "zuzana", "zuza"},
    "VERONICA": {"veronica", "veronika", "vera"},
    "MAGDALENA": {"magdalena", "magdolna", "magdaléna", "magda"},
    "HELENA": {"helena", "ilona", "elena", "lenka"},
    "EVA": {"eva", "éva"},
    "BARBARA": {"barbara", "borbála", "borbala", "barbora"},
    "MARGARETA": {"margaritha", "margaretha", "margit", "margita", "gita"},
    "THERESIA": {"theresia", "teréz", "terez", "terézia", "terezia"},
    "ROSALIA": {"rosalia", "rozália", "rozalia", "róza"},
    "SOPHIA": {"sophia", "zsófia", "zsofia", "žofia", "zofia"},
    "JULIANNA": {"julianna", "juliana", "júlia", "julia", "juliska"},
    "JOHANNA": {"johanna", "jana", "janka", "jolan", "jolana"},
    "LUDOVICA": {"ludovica", "lujza", "ľudmila", "ludmila"},
    "FRANCISCA": {"francisca", "ferike", "františka", "frantiska"},
    "JOSEPHA": {"josepha", "jozefa", "jozefína", "jozefina"},
    "IRENA": {"irena", "irén", "irene"},
    "VINCENCIA": {"vincencia", "vincentia"},
}

_GIVEN_TO_CANONICAL_ID: dict[str, str] = {}
for _cid, _variants in GIVEN_NAME_EQUIVALENCE_GROUPS.items():
    for _v in _variants:
        _GIVEN_TO_CANONICAL_ID[_v] = _cid
        _GIVEN_TO_CANONICAL_ID[_v.translate(_DIACRITIC_FOLD).lower()] = _cid
for _ocr_g, _rep_g in GIVEN_NAME_OCR_REPAIRS.items():
    _rep_low = _rep_g.lower()
    _rep_fold = _rep_low.translate(_DIACRITIC_FOLD).lower()
    _mapped_cid = _GIVEN_TO_CANONICAL_ID.get(_rep_low) or _GIVEN_TO_CANONICAL_ID.get(_rep_fold)
    if _mapped_cid:
        _GIVEN_TO_CANONICAL_ID[_ocr_g] = _mapped_cid
        _GIVEN_TO_CANONICAL_ID[_ocr_g.translate(_DIACRITIC_FOLD).lower()] = _mapped_cid


@lru_cache(maxsize=65536)
def fold_text_diacritics(text: str | None) -> str:
    """Folds Slovak, Hungarian, and German diacritics to lowercase ASCII."""
    if not text:
        return ""
    return text.translate(_DIACRITIC_FOLD).lower().strip()


ascii_fold = fold_text_diacritics


def resolve_given_name_canonical_id(given_name: str | None) -> str | None:
    """Resolves a Latin, Hungarian, Slovak, or German given name to its canonical cross-language concept ID."""
    if not given_name:
        return None
    parts = [
        re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]", "", p).lower()
        for p in given_name.split()
    ]
    parts = [p for p in parts if p]
    if not parts:
        return None
    first = parts[0]
    if first in _GIVEN_TO_CANONICAL_ID:
        return _GIVEN_TO_CANONICAL_ID[first]
    folded = fold_text_diacritics(first)
    if folded in _GIVEN_TO_CANONICAL_ID:
        return _GIVEN_TO_CANONICAL_ID[folded]
    return folded.upper()


def get_given_name_equivalents(given_name: str | None) -> list[str]:
    """Returns all cross-language equivalent tokens for a given name (for FTS5 search expansion)."""
    cid = resolve_given_name_canonical_id(given_name)
    if not cid:
        return []
    group = GIVEN_NAME_EQUIVALENCE_GROUPS.get(cid)
    if not group:
        return [fold_text_diacritics(given_name)]
    out: set[str] = set()
    for v in group:
        out.add(v)
        out.add(fold_text_diacritics(v))
    return sorted(out)


expand_given_name_equivalents = get_given_name_equivalents


SLOVAK_ADJECTIVAL_SURNAME_STEMS: dict[str, tuple[str, str]] = {
    "stastn": ("Šťastný", "Šťastná"),
    "stasntn": ("Šťastný", "Šťastná"),
    "stasn": ("Šťastný", "Šťastná"),
    "siastn": ("Šťastný", "Šťastná"),
    "srastn": ("Šťastný", "Šťastná"),
    "rastn": ("Šťastný", "Šťastná"),
    "hastn": ("Šťastný", "Šťastná"),
    "hastu": ("Šťastný", "Šťastná"),
    "opatrn": ("Opatrný", "Opatrná"),
    "opacit": ("Opacitý", "Opacitá"),
    "novotn": ("Novotný", "Novotná"),
    "vesel": ("Veselý", "Veselá"),
    "pokorn": ("Pokorný", "Pokorná"),
    "ciern": ("Čierny", "Čierna"),
    "cern": ("Černý", "Černá"),
    "biel": ("Biely", "Biela"),
    "mal": ("Malý", "Malá"),
    "nov": ("Nový", "Nová"),
    "star": ("Starý", "Stará"),
    "dlh": ("Dlhý", "Dlhá"),
    "kratk": ("Krátky", "Krátka"),
    "such": ("Suchý", "Suchá"),
    "tich": ("Tichý", "Tichá"),
    "mudr": ("Múdry", "Múdra"),
    "pekn": ("Pekný", "Pekná"),
    "krasn": ("Krásny", "Krásna"),
    "konecn": ("Konečný", "Konečná"),
    "slobodn": ("Slobodný", "Slobodná"),
    "dobr": ("Dobrý", "Dobrá"),
    "hrd": ("Hrdý", "Hrdá"),
    "siv": ("Sivý", "Sivá"),
    "piln": ("Pilný", "Pilná"),
    "zlat": ("Zlatý", "Zlatá"),
    "zlt": ("Žltý", "Žltá"),
    "bohat": ("Bohatý", "Bohatá"),
    "chud": ("Chudý", "Chudá"),
    "mlad": ("Mladý", "Mladá"),
    "tvrd": ("Tvrdý", "Tvrdá"),
    "hluch": ("Hluchý", "Hluchá"),
    "plach": ("Plachý", "Plachá"),
    "sladk": ("Sladký", "Sladká"),
    "hladk": ("Hladký", "Hladká"),
    "prudk": ("Prudký", "Prudká"),
    "tazk": ("Ťažký", "Ťažká"),
    "vysok": ("Vysoký", "Vysoká"),
    "sirok": ("Široký", "Široká"),
    "hlbok": ("Hlboký", "Hlboká"),
    "velk": ("Veľký", "Veľká"),
    "tenk": ("Tenký", "Tenká"),
    "makk": ("Mäkký", "Mäkká"),
    "chrom": ("Chromý", "Chromá"),
    "slep": ("Slepý", "Slepá"),
    "slab": ("Slabý", "Slabá"),
    "hrub": ("Hrubý", "Hrubá"),
    "rysav": ("Ryšavý", "Ryšavá"),
    "smel": ("Smelý", "Smelá"),
    "zrel": ("Zrelý", "Zrelá"),
    "ostr": ("Ostrý", "Ostrá"),
    "bystr": ("Bystrý", "Bystrá"),
    "mokr": ("Mokrý", "Mokrá"),
    "lav": ("Ľavý", "Ľavá"),
    "prav": ("Pravý", "Pravá"),
    "bystrick": ("Bystrický", "Bystrická"),
    "majersk": ("Majerský", "Majerská"),
    "brestovansk": ("Brestovanský", "Brestovanská"),
    "bukovsk": ("Bukovský", "Bukovská"),
    "benkovsk": ("Benkovský", "Benkovská"),
    "komarnansk": ("Komarňanský", "Komarňanská"),
    "bohumick": ("Bohumický", "Bohumická"),
    "bohunick": ("Bohunický", "Bohunická"),
    "panovsk": ("Pánovský", "Pánovská"),
}

_ADJECTIVAL_ACCENTED_FEM_ENDINGS: tuple[str, ...] = (
    "ská", "cká", "ná", "tá", "dá", "lá", "rá", "vá", "má", "pá", "bá", "há", "chá", "žá", "šá", "čá", "ká",
)
_ADJECTIVAL_ACCENTED_MASC_ENDINGS: tuple[str, ...] = (
    "ský", "cký", "ný", "tý", "dý", "lý", "rý", "vý", "mý", "pý", "bý", "hý", "chý", "žý", "šý", "čý", "ký",
    "ží", "ší", "čí",
)


def is_slovak_adjectival_surname(surname: str | None) -> bool:
    """Returns True if the surname is a Slovak/Czech adjectival surname (masculine -ý/-í/-ský/-cký or feminine -á/-ská/-cká)."""
    if not surname:
        return False
    s = re.sub(r"[\"'’`„”]", "", surname).strip(" .,-;:()/\\")
    if len(s) < 4:
        return False
    low = s.lower()
    low_f = fold_text_diacritics(s)
    if low_f.endswith(("ova", "ora", "ura")):
        low_f_base = low_f[:-3].rstrip("-")
        if low_f_base.endswith(("y", "a", "i")) and low_f_base[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return True
        if low_f_base in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return True
    if low_f.endswith(("y", "a", "i")) and low_f[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        return True
    if low_f in ("stasnij", "stastny", "stastna", "stasntna", "siastna", "srastna", "rastna", "hastua"):
        return True
    if low.endswith(("ský", "sky", "ská", "ska", "cký", "cky", "cká", "cka")):
        return True
    if low.endswith(("ý", "í")):
        return True
    if low.endswith(_ADJECTIVAL_ACCENTED_FEM_ENDINGS) and not low.endswith(("ová", "orá", "urá")):
        return True
    return False


def feminize_slovak_adjectival_surname(surname: str | None) -> str | None:
    """
    If `surname` is a Slovak/Czech adjectival surname (e.g. Šťastný, Veselý, Čierny, Krátky, Bystrický, Majerský),
    returns its proper feminine adjectival form (-á / -a / -ská / -cká, e.g. Šťastná, Čierna, Krátka, Bystrická).
    Returns None if `surname` is NOT an adjectival surname.
    NEVER appends -ová!
    """
    if not surname:
        return None
    s = re.sub(r"[\"'’`„”]", "", surname).strip(" .,-;:()/\\")
    if len(s) < 4:
        return None
    low = s.lower()
    low_f = fold_text_diacritics(s)
    # Strip any corrupted trailing -ová/-ova first if attached to an adjectival stem
    if low.endswith(("ová", "ova", "orá", "ora")) and len(s) >= 6:
        s_pre = s[:-3].rstrip("-")
        low_pre = s_pre.lower()
        low_f_pre = fold_text_diacritics(s_pre)
        if low_f_pre in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f_pre][1]
        if low_f_pre.endswith(("y", "a", "i")) and low_f_pre[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f_pre[:-1]][1]
        if low_pre.endswith(("ský", "sky", "ská", "ska", "cký", "cky", "cká", "cka", "ý", "í", "á")):
            s = s_pre
            low = low_pre
            low_f = low_f_pre

    if low_f in ("stasnij", "stasny", "stastny", "stastna", "stasntna", "siastna", "srastna", "rastna", "hastua", "hastu"):
        return "Šťastná"
    if low_f in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f][1]
    if low_f.endswith(("y", "a", "i")) and low_f[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        masc, fem = SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f[:-1]]
        # Preserve original diacritics on prefix if present (e.g. Opačitý -> Opačitá)
        if s[:-1].lower() == masc[:-1].lower() and s[:-1] != masc[:-1]:
            return s[:-1] + fem[-1]
        return fem
    if low.endswith(("ská", "cká")):
        return s[0].upper() + s[1:]
    if low.endswith(("ský", "sky", "ska")):
        return s[0].upper() + s[1:-3] + "ská"
    if low.endswith(("cký", "cky", "cka")):
        return s[0].upper() + s[1:-3] + "cká"
    if low.endswith(_ADJECTIVAL_ACCENTED_FEM_ENDINGS) and not low.endswith(("ová", "orá", "urá")):
        return s[0].upper() + s[1:]
    if low.endswith(("ý", "í")):
        return s[0].upper() + s[1:-1] + "á"
    return None


def masculinize_slovak_adjectival_surname(surname: str | None) -> str | None:
    """
    If `surname` is a Slovak/Czech adjectival surname (masculine or feminine, e.g. Šťastná, Bystrická, Čierna),
    returns its canonical masculine adjectival stem (e.g. Šťastný, Bystrický, Čierny).
    Returns None if `surname` is NOT an adjectival surname.
    """
    if not surname:
        return None
    s = re.sub(r"[\"'’`„”]", "", surname).strip(" .,-;:()/\\")
    if len(s) < 4:
        return None
    low = s.lower()
    low_f = fold_text_diacritics(s)
    if low.endswith(("ová", "ova", "orá", "ora")) and len(s) >= 6:
        s_pre = s[:-3].rstrip("-")
        low_pre = s_pre.lower()
        low_f_pre = fold_text_diacritics(s_pre)
        if low_f_pre in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f_pre][0]
        if low_f_pre.endswith(("y", "a", "i")) and low_f_pre[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f_pre[:-1]][0]
        if low_pre.endswith(("ský", "sky", "ská", "ska", "cký", "cky", "cká", "cka", "ý", "í", "á")):
            s = s_pre
            low = low_pre
            low_f = low_f_pre

    if low_f in ("stasnij", "stasny", "stastny", "stastna", "stasntna", "siastna", "srastna", "rastna", "hastua", "hastu"):
        return "Šťastný"
    if low_f in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        return SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f][0]
    if low_f.endswith(("y", "a", "i")) and low_f[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        masc, _ = SLOVAK_ADJECTIVAL_SURNAME_STEMS[low_f[:-1]]
        if s[:-1].lower() == masc[:-1].lower() and s[:-1] != masc[:-1]:
            return s[:-1] + masc[-1]
        return masc
    if low.endswith(("ský", "cký")):
        return s[0].upper() + s[1:]
    if low.endswith(("ská", "ska", "sky")):
        return s[0].upper() + s[1:-3] + "ský"
    if low.endswith(("cká", "cka", "cky")):
        return s[0].upper() + s[1:-3] + "cký"
    if low.endswith(_ADJECTIVAL_ACCENTED_FEM_ENDINGS) and not low.endswith(("ová", "orá", "urá")):
        return s[0].upper() + s[1:-1] + "ý"
    if low.endswith(("ý", "í")):
        return s[0].upper() + s[1:]
    return None


def sanitize_corrupted_surname_suffix(surname: str | None) -> str | None:
    """
    Repairs corrupted surname suffixes such as:
    - Double feminine suffixes: *ováová, *ovaova -> *ová
    - Forcefully appended -ová on adjectival surnames:
      Šťastnaová / Šťastnáová / Stastnaova / Stasntnaova / Šťastnýová / Stastnyova -> Šťastná
      Bystrickáová / Bystrickýová -> Bystrická
      Majerskáová / Majerskýová -> Majerská
    - Hyphenated suffixes: Buchta-ová / Buchta-ov' -> Buchtová, Buchta- -> Buchta
    - Unstripped vowel before -ová on noun stems:
      Parabaová -> Parabová, Škodaová -> Škodová, Liškaová -> Lišková, Vančoová -> Vančová
    - OCR misreads of Šťastná: Stasntna, Řastná, Siastnová, Haštuová, Hastuová -> Šťastná
    """
    if not surname:
        return None
    s = re.sub(r"[\"'’`„”]", "", surname).strip(" .,;:()/\\")
    for src_ch, dst_ch in (("Ś", "Š"), ("ś", "š"), ("Ć", "Č"), ("ć", "č"), ("Đ", "D"), ("đ", "d"), ("Ł", "L"), ("ł", "l")):
        if src_ch in s:
            s = s.replace(src_ch, dst_ch)
    s = s.rstrip("-")
    if not s:
        return None

    # Collapse repeated -ová / -ova suffixes (e.g. Kováčováová -> Kováčová)
    while re.search(r"(?:ová|ova)(?:ová|ova)$", s, flags=re.IGNORECASE) and len(s) >= 7:
        s = re.sub(r"(?:ová|ova)(?:ová|ova)$", "ová", s, flags=re.IGNORECASE)

    # Fix hyphenated -ová / -ova / -ov (e.g. Buchta-ová -> Buchtová, Lambal-ová -> Lambalová)
    m_hyph = re.match(r"^(.+?)-(?:ová|ova|ov)$", s, flags=re.IGNORECASE)
    if m_hyph:
        base_h = m_hyph.group(1).rstrip("-")
        if base_h.lower().endswith(("a", "o")) and len(base_h) >= 4:
            s = base_h[:-1] + "ová"
        elif len(base_h) >= 3:
            s = base_h + "ová"

    low = s.lower()
    low_f = fold_text_diacritics(s)

    # Direct OCR corruptions of Šťastná / Šťastný
    if low_f in ("stasntna", "stastnaova", "stasntnaova", "stastnyova", "siastnova", "siastna", "srastna", "rastna", "hastua", "hastuova"):
        return "Šťastná"
    if low_f in ("stasnij", "stasny", "srastny"):
        return "Šťastný"

    # Check if -ová/-ova was tacked onto an adjectival surname (*skáová, *skýová, *áová, *aová, *ýová, *yová, *íová, *iová)
    if low.endswith(("ová", "ova")) and len(s) >= 6:
        before_ova = s[:-3].rstrip("-")
        before_low = before_ova.lower()
        before_f = fold_text_diacritics(before_ova)
        if before_f in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[before_f][1]
        if before_f.endswith(("y", "a", "i")) and before_f[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
            return SLOVAK_ADJECTIVAL_SURNAME_STEMS[before_f[:-1]][1]
        if before_low.endswith(("ská", "ska", "ský", "sky")) and len(before_ova) >= 5:
            return before_ova[0].upper() + before_ova[1:-3] + "ská"
        if before_low.endswith(("cká", "cka", "cký", "cky")) and len(before_ova) >= 5:
            return before_ova[0].upper() + before_ova[1:-3] + "cká"
        if before_low.endswith(("ý", "í", "á")) and len(before_ova) >= 4:
            return before_ova[0].upper() + before_ova[1:-1] + "á"
        if before_low.endswith(("a", "o")) and len(before_ova) >= 4:
            # Noun stem ending in -a or -o where the final vowel wasn't dropped before -ová (e.g. Parabaová -> Parabová, Vančoová -> Vančová)
            return before_ova[0].upper() + before_ova[1:-1] + "ová"

    return s


# Curated Slovak surname lexicon for Byt surname consensus & paleographic disambiguation
KNOWN_SLOVAK_SURNAMES: dict[str, float] = {
    "Kráľ": 10.0,
    "Bystrický": 9.5,
    "Paraba": 10.0,
    "Kovaľ": 9.5,
    "Koval": 8.5,
    "Kabát": 9.5,
    "Kabat": 8.5,
    "Šidlík": 9.5,
    "Šidlik": 8.5,
    "Varga": 10.0,
    "Horváth": 10.0,
    "Horvát": 9.0,
    "Kováč": 10.0,
    "Kováčik": 9.5,
    "Tóth": 10.0,
    "Číž": 9.5,
    "Mikuš": 9.5,
    "Brestovanský": 9.5,
    "Polák": 9.5,
    "Sedlák": 9.5,
    "Krajčovič": 9.5,
    "Bartko": 9.5,
    "Novák": 10.0,
    "Baláž": 10.0,
    "Hudec": 9.5,
    "Bielik": 9.5,
    "Králik": 9.5,
    "Moravec": 9.5,
    "Holík": 9.5,
    "Holik": 8.5,
    "Holič": 9.0,
    "Homola": 9.5,
    "Vacula": 9.5,
    "Bódi": 9.0,
    "Bodi": 8.5,
    "Škoda": 9.5,
    "Šťastný": 9.5,
    "Opatrný": 9.0,
    "Šulák": 9.5,
    "Gálik": 9.5,
    "Barta": 9.0,
    "Mikula": 9.0,
    "Havalda": 9.0,
    "Kučera": 9.0,
    "Chalupa": 9.0,
    "Sýkora": 9.0,
    "Procházka": 9.0,
    "Žemlička": 9.0,
    "Hruška": 9.0,
    "Krupka": 9.0,
    "Chovanec": 9.0,
    "Lukáč": 9.0,
    "Molnár": 9.0,
    "Nagy": 9.0,
    "Szabó": 9.0,
    "Nemec": 9.0,
    "Kollár": 9.0,
    "Oravec": 9.0,
    "Ševčík": 9.0,
    "Urban": 9.0,
    "Marko": 9.0,
    "Kubík": 9.0,
    "Tomášik": 9.0,
    "Pavlík": 9.0,
    "Šimko": 9.0,
    "Zeman": 9.0,
    "Rybár": 9.0,
    "Slezák": 9.0,
    "Valach": 9.0,
    "Blažek": 9.0,
    "Masár": 9.0,
    "Kadlec": 9.0,
    "Kovář": 9.0,
    "Buchta": 9.5,
    "Liška": 9.5,
    "Vančo": 9.5,
    "Majerský": 9.5,
    "Bukovský": 9.5,
    "Benkovský": 9.5,
    "Komarňanský": 9.5,
    "Bohumický": 9.5,
    "Pilný": 9.0,
    "Harsányi": 9.5,
    "Vinter": 9.5,
    "Hábel": 9.5,
    "Danihel": 9.5,
    "Fabian": 9.5,
    "Mihalík": 9.5,
    "Gábor": 9.5,
    "Sochor": 9.5,
    "Ďuriš": 9.5,
    "Vražič": 9.5,
    "Náhlik": 9.5,
    "Holomek": 9.5,
    "Zobek": 9.5,
    "Jobek": 9.5,
}

_KNOWN_SURNAME_BY_FOLDED: dict[str, tuple[str, float]] = {}
for _k_sur, _w_sur in KNOWN_SLOVAK_SURNAMES.items():
    _f_sur = ascii_fold(_k_sur)
    if _f_sur not in _KNOWN_SURNAME_BY_FOLDED or _w_sur > _KNOWN_SURNAME_BY_FOLDED[_f_sur][1]:
        _KNOWN_SURNAME_BY_FOLDED[_f_sur] = (_k_sur, _w_sur)

_SURNAME_STEM_A_FOLDED: dict[str, str] = {
    ascii_fold(_k_sur): _k_sur
    for _k_sur in KNOWN_SLOVAK_SURNAMES
    if _k_sur.lower().endswith("a") and not _k_sur.lower().endswith(("ová", "ova", "ská", "ska", "cká", "cka", "ná", "na"))
}

_SURNAME_STEM_O_FOLDED: dict[str, str] = {
    ascii_fold(_k_sur): _k_sur
    for _k_sur in KNOWN_SLOVAK_SURNAMES
    if _k_sur.lower().endswith(("o", "ó"))
}

_CENSUS_DITTO_STEM_RE = re.compile(
    r'^(?://+|"+|„+|”+|-\s*["„”//]+\s*-|—\s*["„”//]+\s*—|dtto\.?|detto\.?|taktiež|taktiez|idem|id\.?)$',
    re.IGNORECASE,
)


def strip_slovak_feminine_suffix(surname: str | None) -> str | None:
    """
    Normalizes a Slovak/Czech/Hungarian surname to its canonical masculine stem:
    - Strips quotes/apostrophes (`Haurelova'` -> `Haurel`), maiden qualifiers (`rod. ...`),
      and OCR feminine endings (`-ová`, `-ova`, `-ora`, `-orá`, `-ská`, `-cká`, `-ná`).
    - Recognizes masculine/feminine adjectival pairs (`Šťastný` <-> `Šťastná`, `Majerský` <-> `Majerská`,
      `Bystrický` <-> `Bystrická`) as sharing the exact same canonical stem.
    - Restores final `-a` / `-o` for `-a` / `-o` stems (`Parabová` -> `Paraba`, `Buchtová` -> `Buchta`,
      `Lišková` -> `Liška`, `Vančová` -> `Vančo`).
    - Normalizes foreign OCR lookalike diacritics (`Śkoda` -> `Škoda`, `Skođova` -> `Škoda`).
    """
    if not surname:
        return None
    s = str(surname).strip()
    if not s or _CENSUS_DITTO_STEM_RE.match(re.sub(r"\s+", " ", s)) or s in ('"', "„", "”", "//", "///", '-"-', '- " -', "—\"—", "- // -"):
        return None
    m_sur = re.search(
        r"^(.+?)\s*(?:\(|,|\b)\s*(?:rod\.?|r\.?|rodená|szül\.?|née|geb\.?)\s+[A-Za-záéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ\-]+",
        s,
        re.IGNORECASE,
    )
    if m_sur and m_sur.group(1).strip(" (),.;-"):
        s = m_sur.group(1).strip(" (),.;-")
    s = re.sub(r"^(?:vd\.?|va\.?|vdova|ovd\.?)\s+", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"['’`\"„”]+", "", s).strip(" .,;:()/\\")
    if not s:
        return None
    char_map = str.maketrans({
        "Ś": "Š",
        "ś": "š",
        "Ć": "Č",
        "ć": "č",
        "Đ": "D",
        "đ": "d",
        "Ł": "L",
        "ł": "l",
    })
    s = s.translate(char_map)
    s = sanitize_corrupted_surname_suffix(s) or s
    s = s.rstrip("-")
    if not s:
        return None
    if s[0].islower():
        s = s[0].upper() + s[1:]

    low = s.lower()
    low_f = ascii_fold(low)
    if low_f in ("vancouver", "vancouverova", "vancora", "vaniora", "vanicova", "vanic", "vanicu", "vamico", "vanco", "vancova"):
        return "Vančo"
    if low_f in ("buchstova", "buchst", "buchlova", "buchlov", "buchtova", "buchta"):
        return "Buchta"
    if low_f in ("stasny", "stasnij", "stastny", "stastna", "stasntna", "siastna", "siastnova", "srastna", "srastny", "rastna", "hastua", "hastuova", "hastu"):
        return "Šťastný"

    if low_f.endswith("aura") and len(low_f) >= 6:
        cand_an = low_f[:-4] + "an"
        if cand_an in _KNOWN_SURNAME_BY_FOLDED:
            return _KNOWN_SURNAME_BY_FOLDED[cand_an][0]
    if low_f.endswith(("nha", "nh", "nhova")) and len(low_f) >= 5:
        prefix = re.sub(r"(?:nhova|nha|nh)$", "", low_f)
        if (prefix + "nsky") in _KNOWN_SURNAME_BY_FOLDED:
            return _KNOWN_SURNAME_BY_FOLDED[prefix + "nsky"][0]

    for suf in ("ová", "ova", "ovà", "ovä", "ové", "ove", "ovra", "orá", "ora", "urá", "ura"):
        if low.endswith(suf) and len(s) > len(suf) + 2 and not low.endswith(("bora", "dora", "kora")):
            if suf in ("ora", "ura") and low_f in _SURNAME_STEM_A_FOLDED:
                return _SURNAME_STEM_A_FOLDED[low_f]
            base = s[: -len(suf)].rstrip("-")
            base_f = ascii_fold(base)
            if (base_f + "a") in _SURNAME_STEM_A_FOLDED:
                return _SURNAME_STEM_A_FOLDED[base_f + "a"]
            if (base_f + "o") in _SURNAME_STEM_O_FOLDED:
                return _SURNAME_STEM_O_FOLDED[base_f + "o"]
            if base_f in _KNOWN_SURNAME_BY_FOLDED:
                return _KNOWN_SURNAME_BY_FOLDED[base_f][0]
            for cand in (base + "a", base + "o", base):
                if cand in _DEFAULT_SURNAMES or cand.capitalize() in _DEFAULT_SURNAMES:
                    return cand[0].upper() + cand[1:]
            return base[0].upper() + base[1:] if base else base

    adj_masc = masculinize_slovak_adjectival_surname(s)
    if adj_masc:
        adj_f = ascii_fold(adj_masc)
        if adj_f in _KNOWN_SURNAME_BY_FOLDED:
            return _KNOWN_SURNAME_BY_FOLDED[adj_f][0]
        return adj_masc

    if low_f in _KNOWN_SURNAME_BY_FOLDED:
        return _KNOWN_SURNAME_BY_FOLDED[low_f][0]

    return s


def owner_surname_folded_from_name(owner: str | None) -> str | None:
    """Extracts the last token of an owner's full name, strips any feminine suffix, and returns ascii_fold(last_tok)."""
    if not owner:
        return None
    toks = [t.strip(".,;:()") for t in owner.split() if t.strip(".,;:()")]
    if not toks:
        return None
    last_tok = strip_slovak_feminine_suffix(toks[-1]) or toks[-1]
    return ascii_fold(last_tok)


@lru_cache(maxsize=65536)
def orthographic_fold(token: str | None) -> str:
    """
    Folds Slovak, Hungarian, German, and Latin surname/given-name orthography
    into a canonical stem for OCR-tolerant matching:
    - Diacritic stripping + lowercase
    - cz / cs / ts / tz -> c
    - sz -> s, zs -> z
    - gy / dy -> d, ny / nj -> n, ly / lj -> l
    - w -> v, y -> i, ph -> f, th -> t, gh -> g, sch -> s
    - Double-consonant collapsing (tt -> t, ll -> l, nn -> n, ss -> s, pp -> p, etc.)
    """
    if not token:
        return ""
    s = fold_text_diacritics(token)
    s = re.sub(r"[^a-z0-9]", "", s)
    if not s:
        return ""
    for src, dst in (
        ("sch", "s"),
        ("cz", "c"),
        ("cs", "c"),
        ("ts", "c"),
        ("tz", "c"),
        ("sz", "s"),
        ("zs", "z"),
        ("gy", "d"),
        ("dy", "d"),
        ("ny", "n"),
        ("nj", "n"),
        ("ly", "l"),
        ("lj", "l"),
        ("ph", "f"),
        ("th", "t"),
        ("gh", "g"),
        ("w", "v"),
        ("y", "i"),
    ):
        s = s.replace(src, dst)
    s = re.sub(r"([a-z])\1+", r"\1", s)
    return s


# 2-to-1 / 1-to-2 OCR digraph confusion costs for Kurrent / cursive / typewritten scans
_DIGRAPH_TO_CHAR_COST: dict[tuple[str, str], float] = {
    ("rn", "m"): 0.12,
    ("in", "m"): 0.15,
    ("ni", "m"): 0.15,
    ("nn", "m"): 0.15,
    ("uu", "w"): 0.15,
    ("uu", "v"): 0.15,
    ("vv", "w"): 0.10,
    ("vv", "v"): 0.10,
    ("st", "h"): 0.18,
    ("si", "h"): 0.20,
    ("cl", "d"): 0.15,
    ("li", "h"): 0.18,
    ("li", "u"): 0.18,
    ("ri", "n"): 0.18,
    ("ri", "u"): 0.18,
    ("cz", "c"): 0.05,
    ("cs", "c"): 0.05,
    ("ts", "c"): 0.05,
    ("tz", "c"): 0.05,
    ("sz", "s"): 0.05,
    ("zs", "z"): 0.05,
    ("th", "t"): 0.05,
    ("ph", "f"): 0.05,
    ("gh", "g"): 0.05,
    ("gy", "d"): 0.08,
    ("ny", "n"): 0.08,
    ("ly", "l"): 0.08,
}

_DIGRAPH_TO_DIGRAPH_COST: dict[frozenset[str], float] = {
    frozenset({"ur", "nz"}): 0.12,
    frozenset({"st", "si"}): 0.12,
    frozenset({"st", "sr"}): 0.15,
}


@lru_cache(maxsize=65536)
def _ocr_dp_distance(a: str, b: str) -> float:
    """Weighted edit distance supporting 1-char and 2-char OCR/cursive confusions."""
    if a == b:
        return 0.0
    n, m = len(a), len(b)
    if n == 0:
        return float(m) * 0.78
    if m == 0:
        return float(n) * 0.78

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * 0.78
    for j in range(1, m + 1):
        dp[0][j] = j * 0.78

    for i in range(1, n + 1):
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            if ca == cb:
                sub = 0.0
            else:
                conf = _SEARCH_CONFUSION_COST.get(frozenset({ca, cb}))
                if conf is not None:
                    sub = conf
                elif i == 1 and j == 1:
                    sub = 1.25
                else:
                    sub = 0.95

            best = min(
                dp[i - 1][j] + 0.78,
                dp[i][j - 1] + 0.78,
                dp[i - 1][j - 1] + sub,
            )

            # 2-to-1: a[i-2:i] -> b[j-1]
            if i >= 2:
                dg_a = a[i - 2 : i]
                c21 = _DIGRAPH_TO_CHAR_COST.get((dg_a, cb))
                if c21 is not None:
                    best = min(best, dp[i - 2][j - 1] + c21)

            # 1-to-2: a[i-1] -> b[j-2:j]
            if j >= 2:
                dg_b = b[j - 2 : j]
                c12 = _DIGRAPH_TO_CHAR_COST.get((dg_b, ca))
                if c12 is not None:
                    best = min(best, dp[i - 1][j - 2] + c12)

            # 2-to-2: a[i-2:i] <-> b[j-2:j]
            if i >= 2 and j >= 2:
                dg_a = a[i - 2 : i]
                dg_b = b[j - 2 : j]
                if dg_a != dg_b:
                    c22 = _DIGRAPH_TO_DIGRAPH_COST.get(frozenset({dg_a, dg_b}))
                    if c22 is not None:
                        best = min(best, dp[i - 2][j - 2] + c22)

            dp[i][j] = best
    return dp[n][m]


@lru_cache(maxsize=65536)
def weighted_edit_similarity(left: str | None, right: str | None) -> float:
    """
    Computes OCR confusion-weighted similarity in [0.0, 1.0] between two name tokens.
    Combines diacritic-folded DP distance, Hungarian/German/Slovak orthographic-folded
    DP distance, and Kurrent/cursive digraph confusions.
    """
    if not left or not right:
        return 0.0
    a_raw = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(left))
    b_raw = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(right))
    if not a_raw or not b_raw:
        return 0.0
    if a_raw == b_raw:
        return 1.0

    a_orth = orthographic_fold(left)
    b_orth = orthographic_fold(right)
    if a_orth and a_orth == b_orth:
        return 0.97

    dist_raw = _ocr_dp_distance(a_raw, b_raw)
    sim_raw = max(0.0, 1.0 - dist_raw / max(len(a_raw), len(b_raw), 1))

    dist_orth = _ocr_dp_distance(a_orth, b_orth)
    sim_orth = max(0.0, 1.0 - dist_orth / max(len(a_orth), len(b_orth), 1))

    return round(max(sim_raw, sim_orth), 4)


@lru_cache(maxsize=65536)
def _surname_core_stem(token: str) -> str:
    """Extracts a gender-neutral surname stem for fuzzy comparison."""
    repaired = SURNAME_OCR_REPAIRS.get(token.lower()) or sanitize_corrupted_surname_suffix(token) or token
    stem = strip_slovak_feminine_suffix(repaired) or repaired
    f = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(stem))
    if f in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        return fold_text_diacritics(SLOVAK_ADJECTIVAL_SURNAME_STEMS[f][0])
    if len(f) >= 5 and f[:-1] in SLOVAK_ADJECTIVAL_SURNAME_STEMS:
        return fold_text_diacritics(SLOVAK_ADJECTIVAL_SURNAME_STEMS[f[:-1]][0])
    return f


@lru_cache(maxsize=65536)
def fuzzy_token_similarity(q_tok: str | None, cand_tok: str | None) -> float:
    """
    Computes OCR-aware fuzzy similarity in [0.0, 1.0] between a single query token
    and a single candidate name token:
    1. Exact / prefix / substring folded match (1.0 / 0.95 / 0.90).
    2. Multilingual given-name canonical equivalence (e.g. Ján == Joannes == János == Jano).
    3. Slovak/Hungarian/German surname stem & feminine suffix normalization + orthographic folding.
    4. Cursive / Kurrent / typewritten OCR confusion-weighted similarity.
    """
    if not q_tok or not cand_tok:
        return 0.0
    q_f = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(q_tok))
    c_f = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(cand_tok))
    if not q_f or not c_f:
        return 0.0

    # 1. Exact / prefix / substring folded match
    if q_f == c_f:
        return 1.0
    if len(q_f) >= 2 and c_f.startswith(q_f):
        return 0.95
    if len(q_f) >= 3 and q_f in c_f:
        return 0.90

    # If query token is purely numeric (e.g., year or house number), require exact/substring match
    if q_f.isdigit() or c_f.isdigit():
        return 0.0

    best = 0.0

    # 2. Multilingual given-name canonical equivalence (+ OCR-tolerant given name matching)
    q_cid = resolve_given_name_canonical_id(q_tok)
    c_cid = resolve_given_name_canonical_id(cand_tok)
    if q_cid and q_cid in GIVEN_NAME_EQUIVALENCE_GROUPS:
        if q_cid == c_cid:
            return 0.96
        for eq in GIVEN_NAME_EQUIVALENCE_GROUPS[q_cid]:
            eq_sim = weighted_edit_similarity(eq, cand_tok)
            if eq_sim >= 0.80:
                best = max(best, round(0.94 * eq_sim, 4))
    elif c_cid and c_cid in GIVEN_NAME_EQUIVALENCE_GROUPS:
        for eq in GIVEN_NAME_EQUIVALENCE_GROUPS[c_cid]:
            eq_sim = weighted_edit_similarity(q_tok, eq)
            if eq_sim >= 0.80:
                best = max(best, round(0.94 * eq_sim, 4))

    # 3. Surname stem & feminine suffix normalization + orthographic folding
    q_stem = _surname_core_stem(q_tok)
    c_stem = _surname_core_stem(cand_tok)
    if q_stem and c_stem:
        if q_stem == c_stem:
            return 0.96
        q_orth = orthographic_fold(q_stem)
        c_orth = orthographic_fold(c_stem)
        if q_orth and q_orth == c_orth:
            return 0.95
        if len(q_orth) >= 3 and (c_orth.startswith(q_orth) or q_orth.startswith(c_orth)):
            ratio = min(len(q_orth), len(c_orth)) / max(len(q_orth), len(c_orth))
            if ratio >= 0.75:
                best = max(best, 0.91)

        # 4. OCR confusion-weighted edit similarity on stems and full tokens
        best = max(
            best,
            weighted_edit_similarity(q_stem, c_stem),
            weighted_edit_similarity(q_tok, cand_tok),
        )
        # Handle dropped final vowel (-a / -o / -y / -i) before feminine -ová on OCR'd stems (e.g. Varga vs Yargová -> varg vs yarg)
        q_root = q_stem.rstrip("aoyi") if len(q_stem) >= 5 else q_stem
        c_root = c_stem.rstrip("aoyi") if len(c_stem) >= 5 else c_stem
        if len(q_root) >= 4 and len(c_root) >= 4 and (q_root != q_stem or c_root != c_stem):
            best = max(best, round(weighted_edit_similarity(q_root, c_root) * 0.97, 4))

    return round(best, 4)


_QUERY_STOP_TOKENS: set[str] = {"rod", "rodena", "r", "nee", "geb", "szul", "ml", "st", "okr", "okres"}


def fuzzy_name_match_score(
    query_text: str | None,
    candidate_texts: Iterable[str | None],
    *,
    threshold: float = 0.76,
) -> float:
    """
    Scores a single-token or multi-token name query (e.g. 'Ján Kováč', 'Kováč Ján', 'Kováč')
    against an iterable of candidate name strings from a record.
    Returns a score in [0.0, 1.0] (or 0.0 if any query token fails to reach `threshold`).
    """
    if not query_text or not query_text.strip():
        return 1.0

    raw_q_tokens = re.findall(
        r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+",
        query_text.strip(),
    )
    q_tokens = [t for t in raw_q_tokens if fold_text_diacritics(t) not in _QUERY_STOP_TOKENS]
    if not q_tokens:
        return 1.0

    cand_tokens: list[str] = []
    full_folded_parts: list[str] = []
    for text in candidate_texts:
        if not text:
            continue
        s = str(text).strip()
        if not s:
            continue
        full_folded_parts.append(fold_text_diacritics(s))
        for tok in re.findall(
            r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+",
            s,
        ):
            if fold_text_diacritics(tok) not in _QUERY_STOP_TOKENS:
                cand_tokens.append(tok)

    if not cand_tokens:
        return 0.0

    q_full_folded = fold_text_diacritics(" ".join(q_tokens))
    full_haystack = " ".join(full_folded_parts)
    if q_full_folded and q_full_folded in full_haystack:
        return 1.0 if q_full_folded == full_haystack.strip() else 0.98

    token_scores: list[float] = []
    unique_cand_tokens = list(dict.fromkeys(cand_tokens))
    for qt in q_tokens:
        best_for_qt = 0.0
        for ct in unique_cand_tokens:
            sim = fuzzy_token_similarity(qt, ct)
            if sim > best_for_qt:
                best_for_qt = sim
                if best_for_qt >= 0.99:
                    break
        if best_for_qt < threshold:
            return 0.0
        token_scores.append(best_for_qt)

    return round(sum(token_scores) / len(token_scores), 4)


def fuzzy_name_matches(
    query_text: str | None,
    candidate_texts: Iterable[str | None],
    *,
    threshold: float = 0.76,
) -> bool:
    """Boolean convenience wrapper around `fuzzy_name_match_score`."""
    if not query_text or not query_text.strip():
        return True
    return fuzzy_name_match_score(query_text, candidate_texts, threshold=threshold) >= threshold


def extract_person_fact_name_candidates(fact: Any, ocr_text: str | None = None) -> list[str]:
    """Extracts all searchable name strings (including `surname_candidates_json` and OCR text) from a `PersonFact`."""
    parts: list[str] = []
    for attr in (
        "given_name_raw",
        "given_name_sk",
        "given_name_hu",
        "surname_raw",
        "surname_normalized",
        "mother_maiden_surname",
        "father_name",
        "mother_name",
        "spouse_name",
    ):
        val = getattr(fact, attr, None)
        if val:
            parts.append(str(val))

    raw_cands = getattr(fact, "surname_candidates_json", None)
    if raw_cands:
        try:
            parsed = json.loads(raw_cands) if isinstance(raw_cands, str) else raw_cands
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str) and item:
                        parts.append(item)
                    elif isinstance(item, dict):
                        for k in ("raw_ocr", "normalized_ocr", "consensus_stem", "resolved_surname"):
                            v = item.get(k)
                            if isinstance(v, str) and v:
                                parts.append(v)
        except Exception:
            pass

    if ocr_text:
        parts.append(ocr_text)
    return parts


def extract_maiden_surname(raw_text: str | None) -> tuple[str | None, str | None]:
    """Splits 'Kováčová rod. Vargová' or 'Kováčová (r. Varga)' into (married_surname, maiden_surname)."""
    if not raw_text:
        return None, None
    m = re.search(
        r"^(.+?)\s*(?:\(|,|\b)\s*(?:rod\.?|r\.?|rodená|szül\.?|née|geb\.?)\s+([A-Za-záéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ\-]+)",
        raw_text.strip(),
        re.IGNORECASE,
    )
    if m:
        married_part = m.group(1).strip(" (),.;-")
        maiden = _clean_name_token(m.group(2))
        return married_part or None, maiden
    return raw_text.strip() or None, None


CENSUS_RELATION_MAP: dict[str, str] = {
    # Head of household
    "hlava domácnosti": "hlava domácnosti",
    "hlava domacnosti": "hlava domácnosti",
    "hlava rodiny": "hlava domácnosti",
    "hlava": "hlava domácnosti",
    "majiteľ bytu": "hlava domácnosti",
    "majitel bytu": "hlava domácnosti",
    "majitel'bytu": "hlava domácnosti",
    "mojitel bytu": "hlava domácnosti",
    "majiteľ domu": "hlava domácnosti",
    "majitel domu": "hlava domácnosti",
    "majitel dom.": "hlava domácnosti",
    "majiteľ dom.": "hlava domácnosti",
    "majitel dom": "hlava domácnosti",
    "majiteľ dom": "hlava domácnosti",
    "maj. bytu": "hlava domácnosti",
    "maj. bytn": "hlava domácnosti",
    "maj. byte": "hlava domácnosti",
    "majitel blytn": "hlava domácnosti",
    "maj. dom.": "hlava domácnosti",
    "prednosta domácnosti": "hlava domácnosti",
    "prednosta domacnosti": "hlava domácnosti",
    "prednosta": "hlava domácnosti",
    "predn. dom.": "hlava domácnosti",
    "pred. dom.": "hlava domácnosti",
    "predu. dom.": "hlava domácnosti",
    "predic dom.": "hlava domácnosti",
    "presidenta dominosti": "hlava domácnosti",
    "manibel bydw": "hlava domácnosti",
    "hosp.": "hlava domácnosti",
    "hosp": "hlava domácnosti",
    "hospodár": "hlava domácnosti",
    "hospodar": "hlava domácnosti",
    # Spouse / Partner
    "manželka": "manželka",
    "manzelka": "manželka",
    "manzeelka": "manželka",
    "maurielka": "manželka",
    "maniella": "manželka",
    "manuela": "manželka",
    "manuel": "manželka",
    "manž.": "manželka",
    "manz.": "manželka",
    "žena": "manželka",
    "zena": "manželka",
    "manžel": "manžel",
    "manzel": "manžel",
    "muž": "manžel",
    "muz": "manžel",
    "družka": "družka",
    "druzka": "družka",
    "konkubína": "družka",
    "konkubina": "družka",
    "druh": "druh",
    # Children, Stepchildren, Adopted, Foster
    "nevlastný syn": "nevlastný syn",
    "nevlastny syn": "nevlastný syn",
    "pastorok": "nevlastný syn",
    "nevlastná dcéra": "nevlastná dcéra",
    "nevlastna dcera": "nevlastná dcéra",
    "pastorkyňa": "nevlastná dcéra",
    "pastorkyna": "nevlastná dcéra",
    "pastornyja": "nevlastná dcéra",
    "dce'ra pastelkjó": "nevlastná dcéra",
    "pastelkjó": "nevlastná dcéra",
    "pastelkjo": "nevlastná dcéra",
    "adoptívny syn": "adoptívny syn",
    "adoptivny syn": "adoptívny syn",
    "adopt. syn": "adoptívny syn",
    "osvojený syn": "adoptívny syn",
    "osvojeny syn": "adoptívny syn",
    "adoptívna dcéra": "adoptívna dcéra",
    "adoptivna dcera": "adoptívna dcéra",
    "adopt. dcéra": "adoptívna dcéra",
    "adopt. dcera": "adoptívna dcéra",
    "osvojená dcéra": "adoptívna dcéra",
    "osvojena dcera": "adoptívna dcéra",
    "chovanec": "chovanec",
    "chovanica": "chovanec",
    "chovanka": "chovanec",
    "chovankyňa": "chovanec",
    "chovankyna": "chovanec",
    "syn": "syn",
    "ryn": "syn",
    "ryu": "syn",
    "dyn": "syn",
    "sin": "syn",
    "dcéra": "dcéra",
    "dcera": "dcéra",
    "dce'ra": "dcéra",
    "deera": "dcéra",
    "diera": "dcéra",
    "de a": "dcéra",
    "deora": "dcéra",
    "devara": "dcéra",
    "dearest": "dcéra",
    "33 deca": "dcéra",
    "33 dexa": "dcéra",
    "deca": "dcéra",
    "dexa": "dcéra",
    # Parents & Parents-in-law & Co-parents-in-law
    "svokra": "svokra",
    "svokor's wife": "svokra",
    "matka manželky": "svokra",
    "matka manzelky": "svokra",
    "matka manžela": "svokra",
    "matka manzela": "svokra",
    "tchyně": "svokra",
    "tchyne": "svokra",
    "testiná": "svokra",
    "testina": "svokra",
    "svokor": "svokor",
    "otec manželky": "svokor",
    "otec manzelky": "svokor",
    "otec manžela": "svokor",
    "otec manzela": "svokor",
    "tchán": "svokor",
    "tchan": "svokor",
    "tesť": "svokor",
    "test": "svokor",
    "svatka": "svatka",
    "svatkyňa": "svatka",
    "svatkyna": "svatka",
    "svat": "svat",
    "stará matka": "stará matka",
    "stara matka": "stará matka",
    "babka": "stará matka",
    "babička": "stará matka",
    "babicka": "stará matka",
    "starká": "stará matka",
    "starka": "stará matka",
    "starý otec": "starý otec",
    "stary otec": "starý otec",
    "dedo": "starý otec",
    "dedko": "starý otec",
    "starký": "starý otec",
    "starky": "starý otec",
    "matka": "matka",
    "mať": "matka",
    "mati": "matka",
    "mamka": "matka",
    "otec": "otec",
    # Son/Daughter-in-law
    "zať": "zať",
    "zat": "zať",
    "nevesta": "nevesta",
    "snacha": "nevesta",
    # Grandchildren & Great-grandchildren
    "pravnuk": "pravnuk",
    "pravnučka": "pravnučka",
    "pravnucka": "pravnučka",
    "vnuk": "vnuk",
    "vnučka": "vnučka",
    "vnucka": "vnučka",
    # Siblings & Siblings-in-law
    "brat": "brat",
    "sestra": "sestra",
    "švagor": "švagor",
    "svagor": "švagor",
    "švagriná": "švagriná",
    "svagrina": "švagriná",
    "švagrová": "švagriná",
    "svagrova": "švagriná",
    "wagriná": "švagriná",
    "wagrina": "švagriná",
    # Aunts, Uncles, Nieces, Nephews, Cousins, Relatives
    "strýko": "strýko",
    "stryko": "strýko",
    "strýc": "strýko",
    "stryc": "strýko",
    "stryná": "stryná",
    "stryna": "stryná",
    "strýčná": "stryná",
    "strycna": "stryná",
    "ujo": "ujo",
    "ujec": "ujo",
    "ujko": "ujo",
    "teta": "teta",
    "tetka": "teta",
    "synovec": "synovec",
    "neter": "neter",
    "netere": "neter",
    "bratranec": "bratranec",
    "sesternica": "sesternica",
    "príbuzný": "príbuzný",
    "pribuzny": "príbuzný",
    "príbuzná": "príbuzná",
    "pribuzna": "príbuzná",
    # Household staff & Apprentices
    "slúžka": "slúžka",
    "sluzka": "slúžka",
    "služobná": "slúžka",
    "sluzobna": "slúžka",
    "slúžtička": "slúžka",
    "sluha": "sluha",
    "čeľadník": "sluha",
    "celadnik": "sluha",
    "paholok": "sluha",
    "služobník": "sluha",
    "sluzobnik": "sluha",
    "gazdiná": "gazdiná",
    "gazdina": "gazdiná",
    "hospodyňa": "gazdiná",
    "hospodyna": "gazdiná",
    "pestúnka": "pestúnka",
    "pestunka": "pestúnka",
    "opatrovateľka": "pestúnka",
    "učeník": "učeník",
    "ucenik": "učeník",
    "učeň": "učeník",
    "ucen": "učeník",
    "tovariš": "tovariš",
    "tovaris": "tovariš",
    "pomocník": "pomocník",
    "pomocnik": "pomocník",
    "pomocnica": "pomocník",
    "pomocníčka": "pomocník",
    # Lodgers & Guests
    "podnájomníčka": "podnájomníčka",
    "podnajomnicka": "podnájomníčka",
    "nájomníčka": "podnájomníčka",
    "najomnicka": "podnájomníčka",
    "podnájomník": "podnájomník",
    "podnajomnik": "podnájomník",
    "nájomník": "podnájomník",
    "najomnik": "podnájomník",
    "bývajúci": "bývajúci",
    "byvajuci": "bývajúci",
    "spolubývajúci": "bývajúci",
    "spolubyvajuci": "bývajúci",
    "nocľažník": "bývajúci",
    "noclaznik": "bývajúci",
    "hosť": "hosť",
    "host": "hosť",
    "návšteva": "hosť",
}

_SORTED_RELATION_ITEMS: list[tuple[str, str]] = sorted(
    CENSUS_RELATION_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)
_COMPILED_RELATION_ITEMS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(rf"(?:^|(?<!\w)){re.escape(k.rstrip('.'))}\.?(?:$|(?=\s|[.,;:/()\-'\"’]|$))"),
        v,
    )
    for k, v in _SORTED_RELATION_ITEMS
]


def normalize_census_relation(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    cleaned = re.sub(r"\s+", " ", str(raw_text)).strip()
    if not cleaned:
        return None
    low_raw = cleaned.lower()
    if low_raw in CENSUS_RELATION_MAP:
        return CENSUS_RELATION_MAP[low_raw]
    low = low_raw.strip(".,;:-()")
    if low in CENSUS_RELATION_MAP:
        return CENSUS_RELATION_MAP[low]
    if (low + ".") in CENSUS_RELATION_MAP:
        return CENSUS_RELATION_MAP[low + "."]
    for pat, v in _COMPILED_RELATION_ITEMS:
        if pat.search(low_raw):
            return v
    return cleaned


def parse_house_number_int(house_no: str | None) -> int | None:
    """Extracts the primary integer house number (conscription or orientation number) for indexing."""
    if not house_no:
        return None
    # On Slovakiana, street strings often look like "-/121" or "17/278" (orientačné / súpisné)
    if "/" in house_no:
        parts = [p.strip() for p in house_no.split("/") if p.strip()]
        for p in reversed(parts):
            m = re.search(r"(\d+)", p)
            if m:
                return int(m.group(1))
    m = re.search(r"(\d+)", str(house_no))
    return int(m.group(1)) if m else None


_CANONICAL_MUNICIPALITY_SYNONYMS: dict[str, str] = {
    "horny cepen": "horny cepen",
    "cepen horny": "horny cepen",
    "hor cepen": "horny cepen",
    "hr cepen": "horny cepen",
    "dolny cepen": "dolny cepen",
    "cepen dolny": "dolny cepen",
    "dol cepen": "dolny cepen",
    "dl cepen": "dolny cepen",
    "stredny cepen": "stredny cepen",
    "cepen stredny": "stredny cepen",
    "str cepen": "stredny cepen",
    "velke surovce": "velke surovce",
    "surovce velke": "velke surovce",
    "vel surovce": "velke surovce",
    "male surovce": "male surovce",
    "surovce male": "male surovce",
    "mal surovce": "male surovce",
    "horne zelenice": "horne zelenice",
    "zelenice horne": "horne zelenice",
    "hor zelenice": "horne zelenice",
    "dolne zelenice": "dolne zelenice",
    "zelenice dolne": "dolne zelenice",
    "dol zelenice": "dolne zelenice",
    "velke ripnany": "velke ripnany",
    "ripnany velke": "velke ripnany",
    "vel ripnany": "velke ripnany",
    "male ripnany": "male ripnany",
    "ripnany male": "male ripnany",
    "mal ripnany": "male ripnany",
    "velke topolcany": "velke topolcany",
    "topolcany velke": "velke topolcany",
    "vel topolcany": "velke topolcany",
}

_CANONICAL_MUNICIPALITY_DISPLAY: dict[str, str] = {
    "horny cepen": "Horný Čepeň",
    "dolny cepen": "Dolný Čepeň",
    "stredny cepen": "Stredný Čepeň",
    "velke surovce": "Veľké Šúrovce",
    "male surovce": "Malé Šúrovce",
    "horne zelenice": "Horné Zelenice",
    "dolne zelenice": "Dolné Zelenice",
    "velke ripnany": "Veľké Ripňany",
    "male ripnany": "Malé Ripňany",
    "velke topolcany": "Veľké Topoľčany",
}


@lru_cache(maxsize=16384)
def canonical_municipality_key(municipality: str | None) -> str:
    """
    Deterministic canonical municipality key for strict village-level matching.
    Strips diacritics, lowercases, normalizes punctuation/hyphens, and resolves only
    exact spelling variants of the same village (e.g. 'Horný Čepen' == 'Horný Čepeň').
    NEVER matches distinct villages like 'Horný Čepeň' and 'Horné Zelenice'.
    """
    if not municipality:
        return ""
    folded = fold_text_diacritics(str(municipality))
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if not cleaned:
        return ""
    return _CANONICAL_MUNICIPALITY_SYNONYMS.get(cleaned, cleaned)


def canonical_municipality_display(municipality: str | None) -> str | None:
    """Returns the canonical display name for a municipality (fixing missing diacritics like 'Horný Čepen' -> 'Horný Čepeň')."""
    if not municipality:
        return None
    s = str(municipality).strip()
    if not s:
        return None
    m_key = canonical_municipality_key(s)
    return _CANONICAL_MUNICIPALITY_DISPLAY.get(m_key, s)


def normalize_house_number(house_no: str | int | None) -> str:
    """
    Deterministic normalizer for census house numbers ('Číslo domu' / conscription number).
    Strips prefixes ('č.', 'dom', '-/'), leading zeros ('0026' -> '26'), and punctuation.
    """
    if house_no is None:
        return ""
    raw = str(house_no).strip()
    if not raw:
        return ""
    # Strip common prefixes like 'dom č.', 'č.', 'nro.', 'no.'
    raw = re.sub(
        r"^(?:dom\s*)?(?:č\.?|c\.?|číslo|cislo|nro\.?|no\.?|№|#)\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    if "/" in raw:
        parts = [p.strip(" -.") for p in raw.split("/") if p.strip(" -.")]
        for p in reversed(parts):
            if any(ch.isdigit() for ch in p):
                raw = p
                break
    raw = raw.strip(" .,;-/").lower()
    if not any(ch.isdigit() for ch in raw):
        return ""
    m = re.fullmatch(r"0*(\d+)\s*([a-z]?)", raw)
    if m:
        num_part = str(int(m.group(1)))
        suffix = m.group(2) or ""
        return f"{num_part}{suffix}"
    m_lead = re.search(r"0*(\d+[a-z]?)", raw)
    if m_lead:
        tok = m_lead.group(1)
        m2 = re.fullmatch(r"0*(\d+)([a-z]?)", tok)
        if m2:
            return f"{int(m2.group(1))}{m2.group(2)}"
        return tok
    return ""


_GENERIC_SETTLEMENT_KEYS: set[str] = {
    "",
    "dedina",
    "obec",
    "mlyn",
    "skupina domov",
    "zel strazny domcek",
    "zeleznicny strazny domcek",
    "strazny domcek",
    "dvor",
    "majer",
    "samota",
    "osada",
    "ulica",
}


def is_compatible_settlement(
    overview_settlement: str | None,
    candidate_street: str | None,
    municipality: str | None = None,
) -> bool:
    """
    Checks whether a CensusOverviewHouse settlement_name ('osada') and a CensusSheet street_raw
    are compatible when both specify a sub-settlement/street.
    """
    ov_key = canonical_municipality_key(overview_settlement)
    st_key = canonical_municipality_key(candidate_street)
    muni_key = canonical_municipality_key(municipality)

    if not ov_key or ov_key in _GENERIC_SETTLEMENT_KEYS or (muni_key and ov_key == muni_key):
        return True
    if not st_key or st_key in _GENERIC_SETTLEMENT_KEYS or (muni_key and st_key == muni_key):
        return True
    return ov_key == st_key


def can_match_overview_house_to_sheet(
    ovh: Any,
    candidate_sheet: Any,
    overview_sheet: Any = None,
) -> bool:
    """
    Strict deterministic check enforcing that a CensusOverviewHouse on an overview sheet
    is ONLY ever matched to a household CensusSheet in the exact same canonical municipality,
    same census year, compatible settlement, and same normalized house number.
    """
    if ovh is None or candidate_sheet is None:
        return False
    cand_type = getattr(candidate_sheet, "sheet_type", None) or "household"
    if cand_type == "overview":
        return False

    ov_sheet_id = getattr(ovh, "census_sheet_id", None)
    cand_id = getattr(candidate_sheet, "id", None)
    if ov_sheet_id is not None and cand_id is not None and ov_sheet_id == cand_id:
        return False

    ov_year = getattr(ovh, "census_year", None) or (
        getattr(overview_sheet, "census_year", None) if overview_sheet is not None else None
    )
    cand_year = getattr(candidate_sheet, "census_year", None)
    if ov_year is not None and cand_year is not None and int(ov_year) != int(cand_year):
        return False

    ov_muni_raw = (
        (getattr(overview_sheet, "municipality", None) if overview_sheet is not None else None)
        or getattr(ovh, "municipality", None)
    )
    ov_muni = canonical_municipality_key(ov_muni_raw)
    cand_muni = canonical_municipality_key(getattr(candidate_sheet, "municipality", None))
    if not ov_muni or not cand_muni or ov_muni != cand_muni:
        return False

    if overview_sheet is not None:
        ov_mkey = str(getattr(overview_sheet, "municipality_key", None) or "").strip()
        cand_mkey = str(getattr(candidate_sheet, "municipality_key", None) or "").strip()
        if ov_mkey and cand_mkey and ov_mkey != cand_mkey:
            return False

    ov_hn = normalize_house_number(getattr(ovh, "house_number", None))
    cand_hn = normalize_house_number(getattr(candidate_sheet, "house_number", None))
    if not ov_hn or not cand_hn or ov_hn != cand_hn:
        return False

    if not is_compatible_settlement(
        getattr(ovh, "settlement_name", None),
        getattr(candidate_sheet, "street_raw", None),
        ov_muni_raw,
    ):
        return False

    return True


def compute_person_fact_search_fields(fact_or_dict: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Computes pre-indexed search columns (folded surname, phonetic skeleton, canonical given ID, etc.)."""
    source = kwargs if fact_or_dict is None else fact_or_dict

    def _get(k: str) -> Any:
        if isinstance(source, dict):
            return source.get(k)
        return getattr(source, k, None)

    surname = _get("surname_normalized") or _get("surname_raw")
    surname_stem = strip_slovak_feminine_suffix(surname) or surname
    given = _get("given_name_sk") or _get("given_name_raw") or _get("given_name_hu")
    village = _get("village_sk") or _get("village_raw") or _get("village_hu")
    house_no = _get("house_number")
    birth_year_est = _get("birth_year_est")
    if birth_year_est is None:
        bdate = str(_get("birth_or_event_date") or "")
        ym = re.search(r"\b(1[789]\d{2}|20[0-2]\d)\b", bdate)
        if ym:
            birth_year_est = int(ym.group(1))
        elif _get("record_type") == "baptism" and _get("event_year"):
            birth_year_est = int(_get("event_year"))
        elif _get("event_year") and _get("age_at_event"):
            am = re.search(r"(\d{1,3})", str(_get("age_at_event")))
            if am:
                birth_year_est = int(_get("event_year")) - int(am.group(1))

    return {
        "surname_folded": _apply_digraphs(surname_stem) if surname_stem else None,
        "surname_skeleton": _phonetic_skeleton(surname_stem) if surname_stem else None,
        "given_canonical_id": resolve_given_name_canonical_id(given),
        "village_folded": fold_text_diacritics(village) if village else None,
        "house_number_int": parse_house_number_int(house_no),
        "birth_year_est": birth_year_est,
    }


def apply_person_fact_search_fields(fact: Any) -> dict[str, Any]:
    """Computes search fields for `fact` and assigns the 5 pre-indexed search columns onto `fact`."""
    search_fields = compute_person_fact_search_fields(fact)
    fact.surname_folded = search_fields["surname_folded"]
    fact.surname_skeleton = search_fields["surname_skeleton"]
    fact.given_canonical_id = search_fields["given_canonical_id"]
    fact.village_folded = search_fields["village_folded"]
    fact.house_number_int = search_fields["house_number_int"]
    for extra_col in ("given_name_folded", "house_number_norm", "search_text_folded"):
        if extra_col in search_fields and hasattr(fact, extra_col):
            setattr(fact, extra_col, search_fields[extra_col])
    if getattr(fact, "birth_year_est", None) is None and search_fields.get("birth_year_est") is not None:
        fact.birth_year_est = search_fields["birth_year_est"]
    return search_fields


# ---------------------------------------------------------------------------
# Regional Gazetteer (~100km Vicinity + Slovakiana Facets + FamilySearch Cache)
# ---------------------------------------------------------------------------

_REGIONAL_PLACES_CACHE_LOCK = threading.Lock()
_REGIONAL_PLACES_CACHE: dict[str, Any] | None = None
_SLOVAKIANA_FACETS_LOADED = False
_SLOVAKIANA_DISTRICTS: list[str] = []
_SLOVAKIANA_MUNICIPALITIES: list[str] = []

# Canonical political districts in 1930/1940 Slovakia & neighboring lands
CANONICAL_CENSUS_DISTRICTS: list[str] = [
    "Trnava",
    "Hlohovec",
    "Galanta",
    "Nitra",
    "Topoľčany",
    "Piešťany",
    "Senica",
    "Skalica",
    "Bratislava",
    "Pezinok",
    "Modra",
    "Malacky",
    "Dunajská Streda",
    "Šaľa",
    "Šamorín",
    "Komárno",
    "Nové Zámky",
    "Stará Ďala",
    "Parkáň",
    "Zlaté Moravce",
    "Vráble",
    "Levice",
    "Trenčín",
    "Nové Mesto nad Váhom",
    "Myjava",
    "Bánovce nad Bebravou",
    "Ilava",
    "Púchov",
    "Považská Bystrica",
    "Prievidza",
    "Žilina",
    "Čadca",
    "Kysucké Nové Mesto",
    "Bytča",
    "Turčiansky Svätý Martin",
    "Martin",
    "Ružomberok",
    "Liptovský Svätý Mikuláš",
    "Dolný Kubín",
    "Námestovo",
    "Trstená",
    "Banská Bystrica",
    "Zvolen",
    "Banská Štiavnica",
    "Kremnica",
    "Nová Baňa",
    "Krupina",
    "Lučenec",
    "Modrý Kameň",
    "Rimavská Sobota",
    "Revúca",
    "Rožňava",
    "Brezno",
    "Poprad",
    "Kežmarok",
    "Spišská Nová Ves",
    "Levoča",
    "Gelnica",
    "Stará Ľubovňa",
    "Košice",
    "Prešov",
    "Sabinov",
    "Bardejov",
    "Giraltovce",
    "Stropkov",
    "Svidník",
    "Medzilaborce",
    "Humenné",
    "Snina",
    "Vranov nad Topľou",
    "Michalovce",
    "Trebišov",
    "Sobrance",
    "Kráľovský Chlmec",
]

# Curated ~100km regional gazetteer around Western Slovakia (Trnava / Hlohovec / Galanta / Nitra / Topoľčany)
# Maps canonical municipality/settlement -> (canonical_district, canonical_country)
REGIONAL_GAZETTEER_100KM: dict[str, tuple[str, str]] = {
    "Horný Čepeň": ("Trnava", "Slovensko"),
    "Dolný Čepeň": ("Trnava", "Slovensko"),
    "Stredný Čepeň": ("Trnava", "Slovensko"),
    "Veľké Šúrovce": ("Trnava", "Slovensko"),
    "Malé Šúrovce": ("Trnava", "Slovensko"),
    "Šúrovce": ("Trnava", "Slovensko"),
    "Valtašúr": ("Trnava", "Slovensko"),
    "Varov Šúr": ("Trnava", "Slovensko"),
    "Zemiansky Šúr": ("Trnava", "Slovensko"),
    "Majcichov": ("Trnava", "Slovensko"),
    "Vlčkovce": ("Trnava", "Slovensko"),
    "Opoj": ("Trnava", "Slovensko"),
    "Križovany nad Dudváhom": ("Trnava", "Slovensko"),
    "Zavar": ("Trnava", "Slovensko"),
    "Brestovany": ("Trnava", "Slovensko"),
    "Veľké Brestovany": ("Trnava", "Slovensko"),
    "Malé Brestovany": ("Trnava", "Slovensko"),
    "Bučany": ("Trnava", "Slovensko"),
    "Hrnčiarovce nad Parnou": ("Trnava", "Slovensko"),
    "Hrnčiarovce": ("Trnava", "Slovensko"),
    "Zeleneč": ("Trnava", "Slovensko"),
    "Biely Kostol": ("Trnava", "Slovensko"),
    "Ružindol": ("Trnava", "Slovensko"),
    "Cífer": ("Trnava", "Slovensko"),
    "Voderady": ("Trnava", "Slovensko"),
    "Pavlice": ("Trnava", "Slovensko"),
    "Špačince": ("Trnava", "Slovensko"),
    "Jaslovské Bohunice": ("Trnava", "Slovensko"),
    "Malženice": ("Trnava", "Slovensko"),
    "Dechtice": ("Trnava", "Slovensko"),
    "Smolenice": ("Trnava", "Slovensko"),
    "Boleráz": ("Trnava", "Slovensko"),
    "Suchá nad Parnou": ("Trnava", "Slovensko"),
    "Zvončín": ("Trnava", "Slovensko"),
    "Bohdanovce nad Trnavou": ("Trnava", "Slovensko"),
    "Šelpice": ("Trnava", "Slovensko"),
    "Trnava": ("Trnava", "Slovensko"),
    "Siladice": ("Hlohovec", "Slovensko"),
    "Trakovice": ("Hlohovec", "Slovensko"),
    "Horné Zelenice": ("Hlohovec", "Slovensko"),
    "Dolné Zelenice": ("Hlohovec", "Slovensko"),
    "Dvorníky": ("Hlohovec", "Slovensko"),
    "Bojničky": ("Hlohovec", "Slovensko"),
    "Posádka": ("Hlohovec", "Slovensko"),
    "Sasinkovo": ("Hlohovec", "Slovensko"),
    "Kľačany": ("Hlohovec", "Slovensko"),
    "Pastuchov": ("Hlohovec", "Slovensko"),
    "Hlohovec": ("Hlohovec", "Slovensko"),
    "Leopoldov": ("Hlohovec", "Slovensko"),
    "Červeník": ("Hlohovec", "Slovensko"),
    "Madunice": ("Hlohovec", "Slovensko"),
    "Sereď": ("Galanta", "Slovensko"),
    "Šintava": ("Galanta", "Slovensko"),
    "Šoporňa": ("Galanta", "Slovensko"),
    "Pata": ("Galanta", "Slovensko"),
    "Pusté Sady": ("Galanta", "Slovensko"),
    "Zemianske Sady": ("Galanta", "Slovensko"),
    "Vinohrady nad Váhom": ("Galanta", "Slovensko"),
    "Váhovce": ("Galanta", "Slovensko"),
    "Dolná Streda": ("Galanta", "Slovensko"),
    "Gáň": ("Galanta", "Slovensko"),
    "Galanta": ("Galanta", "Slovensko"),
    "Sládkovičovo": ("Galanta", "Slovensko"),
    "Abrahám": ("Galanta", "Slovensko"),
    "Hoste": ("Galanta", "Slovensko"),
    "Malá Mača": ("Galanta", "Slovensko"),
    "Veľká Mača": ("Galanta", "Slovensko"),
    "Veľké Topoľčany": ("Topoľčany", "Slovensko"),
    "Topoľčany": ("Topoľčany", "Slovensko"),
    "Veľké Ripňany": ("Topoľčany", "Slovensko"),
    "Malé Ripňany": ("Topoľčany", "Slovensko"),
    "Nitra": ("Nitra", "Slovensko"),
    "Báb": ("Nitra", "Slovensko"),
    "Rumanová": ("Nitra", "Slovensko"),
    "Rišňovce": ("Nitra", "Slovensko"),
    "Alekšince": ("Nitra", "Slovensko"),
    "Lukáčovce": ("Nitra", "Slovensko"),
    "Poľný Kesov": ("Nitra", "Slovensko"),
    "Piešťany": ("Piešťany", "Slovensko"),
    "Vrbové": ("Piešťany", "Slovensko"),
    "Drahovce": ("Piešťany", "Slovensko"),
    "Viedeň": ("", "Rakúsko"),
}


def _regional_places_cache_path() -> Path:
    try:
        from backend.app.config import settings

        return Path(settings.data_dir) / "regional_places_cache.json"
    except Exception:
        return Path("data/regional_places_cache.json")


def _load_regional_places_cache() -> dict[str, Any]:
    global _REGIONAL_PLACES_CACHE
    with _REGIONAL_PLACES_CACHE_LOCK:
        if _REGIONAL_PLACES_CACHE is not None:
            return _REGIONAL_PLACES_CACHE
        p = _regional_places_cache_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _REGIONAL_PLACES_CACHE = data
                    return _REGIONAL_PLACES_CACHE
            except Exception:
                pass
        _REGIONAL_PLACES_CACHE = {"places": {}, "queries": {}}
        return _REGIONAL_PLACES_CACHE


def record_regional_places_cache(
    query_key: str,
    places: list[dict[str, Any]],
    default_district: str | None = None,
    vicinity_places: list[dict[str, Any]] | None = None,
) -> None:
    """Persist FamilySearch MCP `search_places_at_date` results (and 100km vicinity places) into `data/regional_places_cache.json`."""
    cache = _load_regional_places_cache()
    with _REGIONAL_PLACES_CACHE_LOCK:
        q_map = cache.setdefault("queries", {})
        p_map = cache.setdefault("places", {})
        q_map[fold_text_diacritics(query_key)] = places
        combined = list(places or []) + list(vicinity_places or [])
        for pl in combined:
            if not isinstance(pl, dict):
                continue
            name = str(pl.get("name") or "").strip().replace("'", "ď" if str(pl.get("name") or "").endswith("d'") else "'")
            full_name = str(pl.get("full_name") or name).strip()
            if not name or name.lower() in ("czechoslovakia", "slovakia", "hungary", "austria"):
                continue
            if any(
                bad in full_name.lower()
                for bad in ("suriname", "colombia", "oman", "türkiye", "turkey", "united states", "brazil")
            ):
                continue
            parts = [pt.strip() for pt in full_name.split(",") if pt.strip()]
            dist = (parts[1] if len(parts) >= 3 else "") or default_district or ""
            cnt = "Slovensko"
            if any(x in full_name.lower() for x in ("moravia", "morava", "třebíč", "znojmo", "brno")):
                cnt = "Morava"
            elif any(x in full_name.lower() for x in ("bohemia", "čechy", "klatovy", "praha")):
                cnt = "Čechy"
            elif any(x in full_name.lower() for x in ("austria", "rakúsko", "wien", "vienna")):
                cnt = "Rakúsko"
            p_map[fold_text_diacritics(name)] = {
                "id": pl.get("id"),
                "name": name,
                "district": dist,
                "country": cnt,
                "full_name": full_name,
                "latitude": pl.get("latitude"),
                "longitude": pl.get("longitude"),
            }
        try:
            p = _regional_places_cache_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def _ensure_slovakiana_facets_loaded() -> None:
    global _SLOVAKIANA_FACETS_LOADED, _SLOVAKIANA_DISTRICTS, _SLOVAKIANA_MUNICIPALITIES
    if _SLOVAKIANA_FACETS_LOADED:
        return
    _SLOVAKIANA_FACETS_LOADED = True
    for candidate in (
        Path("data/slovakiana_facets_all.json"),
        Path("data/uploads/slovakiana/slovakiana_facets_all.json"),
    ):
        if candidate.exists():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    _SLOVAKIANA_DISTRICTS = [
                        str(d["name"]).strip()
                        for d in (raw.get("districts") or [])
                        if isinstance(d, dict) and d.get("name")
                    ]
                    seen_m: set[str] = set()
                    for m in raw.get("municipalities") or []:
                        if isinstance(m, dict) and m.get("name"):
                            nm = str(m["name"]).strip()
                            if nm and nm not in seen_m:
                                seen_m.add(nm)
                                _SLOVAKIANA_MUNICIPALITIES.append(nm)
                break
            except Exception:
                pass


def get_regional_municipality_gazetteer(
    sheet_municipality: str | None = None,
    sheet_district: str | None = None,
) -> list[str]:
    """
    Returns the prioritized list of municipalities and settlements within ~100km of the sheet,
    combining `REGIONAL_GAZETTEER_100KM`, `data/regional_places_cache.json`, and `slovakiana_facets_all.json`.
    """
    _ensure_slovakiana_facets_loaded()
    cache = _load_regional_places_cache()
    result: list[str] = []
    seen: set[str] = set()

    def _add(nm: str | None) -> None:
        if not nm:
            return
        c = nm.strip()
        if c and c not in seen:
            seen.add(c)
            result.append(c)

    if sheet_municipality:
        # Canonicalize Horný Čepen -> Horný Čepeň
        if fold_text_diacritics(sheet_municipality) == "horny cepen":
            _add("Horný Čepeň")
        else:
            _add(sheet_municipality)

    # First add same-district places from REGIONAL_GAZETTEER_100KM
    if sheet_district:
        df = fold_text_diacritics(sheet_district)
        for m_name, (m_dist, _) in REGIONAL_GAZETTEER_100KM.items():
            if fold_text_diacritics(m_dist) == df:
                _add(m_name)

    # Next add all 100km regional gazetteer entries
    for m_name in REGIONAL_GAZETTEER_100KM:
        _add(m_name)

    # Next add cached FamilySearch MCP places
    for info in (cache.get("places") or {}).values():
        if isinstance(info, dict) and info.get("name"):
            _add(str(info["name"]))

    return result


# Non-place tokens that must NEVER be accepted as a municipality/district/country
_NON_PLACE_GARBAGE_TOKENS: set[str] = {
    "zenaty", "fenaty", "ievant", "feurle", "furhe", "jeurhe", "jeurla", "munrle", "murfle",
    "munhe", "inbre", "vydatá", "vydata", "slobodny", "slobodna", "vdova", "vdovec", "rdovec",
    "rozvedeny", "rozvedena", "muzske", "zenske", "muz", "zena", "hlava", "manzelka", "manuela",
    "dcera", "deera", "diera", "syn", "dyn", "svokra", "svagrina", "wagrina", "rolnik", "domacnost",
    "domacuot", "domacno", "pol defautore", "defautore", "trvale", "trvalo", "troale", "twale",
    "dtto", "dtt", "ditto", "taktiez",
}


def _strip_place_rubric_noise(raw: str | None) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    # Remove rubric markers like (a), (b), (c), a), b), c) at start or end
    s = re.sub(r"^\s*(?:\([a-d]\)|[a-d]\))\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*(?:\([a-d]\)|[a-d]\))\s*$", "", s, flags=re.IGNORECASE)
    # Strip citizenship prefixes like "Čsl.", "Čs.", "ČSR", "Cil.", "Ohl.", "CH-"
    s = re.sub(r"^(?:čsl\.?|čs\.?|čsr|cil\.?|ohl\.?|čel\.?|ch-)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*okr(?:es)?\.?\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(" .,;:-/\\'\"`„”")
    return re.sub(r"\s+", " ", s).strip()


def normalize_census_country(
    raw_country: str | None,
    inferred_country: str | None = None,
) -> str | None:
    """
    Normalizes census `c) Krajina` (country / historical crownland) readings.
    Repairs cursive OCR misreads of `"Slovenská"` / `"Slovensko"` such as
    `"Horeuska'"`, `"Cloreusra"`, `"Novenka"`, `"Movenka"`, `"Hovarska"`, `"Moreušá"`,
    `"Sloreurha'"`, `"Slavuria"`, `"Slovenia"`, `"Shreno."`, `"Hlreno."`, `"storeuho"`,
    `"Norwala"`, `"Horvinski"`, `"Hoventri"`.
    """
    if not raw_country:
        return inferred_country
    cleaned = _strip_place_rubric_noise(raw_country)
    if not cleaned or cleaned in ("/", "-", "//", '"'):
        return inferred_country
    f = fold_text_diacritics(cleaned)
    if not f:
        return inferred_country

    if any(k in f for k in ("morav",)):
        return "Morava"
    if any(k in f for k in ("cech", "bohem")):
        return "Čechy"
    if any(k in f for k in ("sliez", "siles")):
        return "Sliezsko"
    if any(k in f for k in ("podkarp", "rusin")):
        return "Podkarpatská Rus"
    if any(k in f for k in ("rakus", "austr", "osterr", "vieden", "wien")):
        return "Rakúsko"
    if any(k in f for k in ("madar", "uhors", "hungar", "magyar")):
        return "Maďarsko"
    if any(k in f for k in ("nemec", "german", "deutsch")):
        return "Nemecko"
    if any(k in f for k in ("polsk", "poland", "halic", "galic")):
        return "Poľsko"
    if any(k in f for k in ("rumun", "roman")):
        return "Rumunsko"
    if any(k in f for k in ("juhosl", "uhoslav", "srbsk", "chorvat")):
        return "Juhoslávia"
    if any(k in f for k in ("amerik", "usa", "spojene staty")):
        return "USA"
    if any(k in f for k in ("kanad", "canad")):
        return "Kanada"
    if any(k in f for k in ("francuz", "france")):
        return "Francúzsko"

    # Cursive OCR variants of "Slovenská" / "Slovensko"
    _slovensko_ocr_markers = (
        "sloven", "slovak", "slov", "slore", "slavr", "slavur", "clore", "horeus", "hovar",
        "hoven", "novenk", "movenk", "moreus", "morenh", "morel", "shren", "hlren", "storeu",
        "norwal", "horvin", "dorens", "dorell", "lovens", "loveus", "stonework", "november",
        "dureun",
    )
    if any(m in f for m in _slovensko_ocr_markers):
        return "Slovensko"

    # Paleographic similarity check against "slovenska" / "slovensko"
    sim_sk = max(
        difflib.SequenceMatcher(None, f, "slovenska").ratio(),
        difflib.SequenceMatcher(None, f, "slovensko").ratio(),
    )
    if sim_sk >= 0.52:
        return "Slovensko"

    return inferred_country or "Slovensko"


def normalize_census_district(
    raw_district: str | None,
    sheet_district: str | None = None,
    inferred_district: str | None = None,
) -> str | None:
    """
    Normalizes census `b) Okres` (political district) readings against `CANONICAL_CENSUS_DISTRICTS`
    and `slovakiana_facets_all.json` districts.
    Repairs cursive OCR misreads of `"Trnava"` (`Twinna`, `Tjunava`, `Tmavea`, `Titava`, `Tjnava`,
    `Tinava`, `Truova`, `Truava`, `Trivava`, `Tjava`, `Tmuva`, `Tjurna`, `Tjuvava`, `Tjmasa`,
    `Tjurnava`, `Tjiva`, `Tjama`, `Tirana`, `Tirava`, `Inava`), `"Hlohovec"` (`Hlohove`, `Hlohoove`,
    `Mohoves`, `Helshore`), `"Nitra"` (`mitra`, `vitra`), etc.
    """
    if not raw_district:
        return inferred_district or sheet_district
    cleaned = _strip_place_rubric_noise(raw_district)
    if not cleaned or cleaned in ("/", "-", "//", '"'):
        return inferred_district or sheet_district
    f = fold_text_diacritics(cleaned)
    if not f or f in _NON_PLACE_GARBAGE_TOKENS:
        return inferred_district or sheet_district

    # If someone wrote "Slovenská" / "Slovensko" in the district slot by mistake, fall back to inferred/sheet district
    if any(m in f for m in ("sloven", "clore", "horeus", "novenk", "movenk", "hovar")):
        return inferred_district or sheet_district

    # Direct paleographic table for common 1930/1940 cursive district OCR misreads
    _trnava_variants = {
        "trnava", "twinna", "tjunava", "tmavea", "titava", "tjnava", "tinava", "truova",
        "truava", "trivava", "tjava", "tmuva", "tjurna", "tjuvava", "tjmasa", "tjurnava",
        "tjiva", "tjama", "tirana", "tirava", "inava", "trnava d", "tinava d", "trnava a",
    }
    if f in _trnava_variants or re.fullmatch(r"t[rjmwvniuao]{2,6}va?", f) or re.fullmatch(r"twi?nn?a", f):
        return "Trnava"

    if f in ("hlohovec", "hlohove", "hlohoove", "mohoves", "helshore", "fraistat", "freistadt") or "hlohov" in f:
        return "Hlohovec"
    if f in ("nitra", "mitra", "vitra", "nyitra"):
        return "Nitra"
    if "topolcan" in f:
        return "Topoľčany"
    if "galant" in f:
        return "Galanta"
    if "piestan" in f:
        return "Piešťany"

    _ensure_slovakiana_facets_loaded()
    all_districts = list(dict.fromkeys(CANONICAL_CENSUS_DISTRICTS + _SLOVAKIANA_DISTRICTS))

    best_dist: str | None = None
    best_score = 0.0
    sheet_df = fold_text_diacritics(sheet_district) if sheet_district else ""
    inferred_df = fold_text_diacritics(inferred_district) if inferred_district else ""

    for cand in all_districts:
        cf = fold_text_diacritics(cand)
        if cf == f:
            return cand
        dist_cost = _paleographic_distance(f, cf)
        max_l = max(len(f), len(cf), 1)
        sim = max(0.0, 1.0 - (dist_cost / max_l))
        seq = difflib.SequenceMatcher(None, f, cf).ratio()
        score = max(sim, seq)
        if sheet_df and cf == sheet_df:
            score += 0.12
        if inferred_df and cf == inferred_df:
            score += 0.14
        if score > best_score:
            best_score = score
            best_dist = cand

    if best_dist and best_score >= 0.64:
        return best_dist

    return inferred_district or sheet_district or cleaned


def _expand_place_abbreviations_and_inversions(text: str) -> str:
    """
    Expands Slovak scribe abbreviations and word-order inversions in place names:
    - `Hor. Čepeň`, `Hr. Čepeň`, `Hrv. Čepen`, `Hw. Cipen`, `Hrr. Cepen`, `H. Cepen`, `Dr. Čepen`,
      `Jepen Horni`, `Čepeň Horní`, `Horczen`, `Čepeu hortni`, `Čepen Koru`, `Čepen Hotni`, `Čepel Horn` -> `Horný Čepeň`
    - `Vel' Surovee`, `Vel. Suvorce`, `Veľ. Šúrovce`, `Survee Vel.`, `Linnee Vell`, `Hel' Eurove` -> `Veľké Šúrovce`
    - `Valtasin`, `Valta' n` -> `Valtašúr`
    """
    s = _strip_place_rubric_noise(text)
    if not s:
        return ""
    f = fold_text_diacritics(s)
    f_compact = re.sub(r"[^a-z0-9]+", " ", f).strip()

    # 1. Direct recognition of Horný Čepeň / Stredný Čepeň / Dolný Čepeň cursive OCR forms
    if any(
        pat in f_compact
        for pat in (
            "cepen", "cepeu", "cepel", "cipen", "jepen", "cyepen", "ceypen",
            "horczen", "hor czen", "egeen", "dr epen", "hr cepen", "hrv cepen",
            "hw cipen", "hrr cepen",
        )
    ):
        if any(st in f_compact for st in ("stred", "str ", "ared")):
            return "Stredný Čepeň"
        if any(dl in f_compact for dl in ("doln", "dol ", "dl ")):
            return "Dolný Čepeň"
        return "Horný Čepeň"

    # 2. Direct recognition of Veľké Šúrovce cursive OCR forms
    if any(
        pat in f_compact
        for pat in (
            "surovc", "surove", "suvorc", "survee", "eurove", "linnee vell",
        )
    ):
        if any(ml in f_compact for ml in ("male", "mal ")):
            return "Malé Šúrovce"
        return "Veľké Šúrovce"

    # 3. Direct recognition of Hrnčiarovce (nad Parnou) cursive OCR forms (`Huciavorce`, `Hrucianoe`, `Hrvatskoe`, `Hrvciaro a`)
    if any(pat in f_compact for pat in ("huciavorc", "hruciano", "hrnciarov", "hrvatskoe", "hrvciaro")):
        return "Hrnčiarovce nad Parnou"

    # 3b. Direct recognition of Trnava (`Thuana`, `Truova`, `Tjunava`, `Tmavea`)
    if f_compact in ("thuana", "trnava", "truova", "tjunava", "tmavea", "tjnava", "tinava"):
        return "Trnava"

    # 4. Direct recognition of Valtašúr (`Valtasin`, `Valta n`, `Valtov Sur`, `Valtasur`)
    if any(pat in f_compact for pat in ("valtasin", "valtasur", "valtov sur", "valta n", "valta s")):
        return "Valtašúr"

    # 5. Direct recognition of Majcichov (`Majicíchov`, `Majrichov`, `Majicihov`)
    if any(pat in f_compact for pat in ("majcich", "majicich", "majrich", "majicih")):
        return "Majcichov"

    # 6. Direct recognition of Siladice (`Silasice`, `Silarice`, `Silavice`, `Vacavii`)
    if any(pat in f_compact for pat in ("siladic", "silasic", "silaric", "silavic")):
        return "Siladice"

    # 7. Direct recognition of Veľké Topoľčany (`Topolčany Velke`, `Velke Topolčany`)
    if "topolcan" in f_compact and "vel" in f_compact:
        return "Veľké Topoľčany"

    # 8. Direct recognition of Viedeň
    if f_compact in ("vieden", "wien", "vienna"):
        return "Viedeň"

    # General prefix expansion (`Hor.`, `Hr.`, `Veľ.`, `Vel.`, `Dol.`, `Mal.`)
    s_exp = re.sub(r"^(?:Hor|Hr|Hrv|Hw|Hrr)\.\s*", "Horný ", s, flags=re.IGNORECASE)
    s_exp = re.sub(r"^(?:Veľ|Vel|Vel'|V)\.\s*", "Veľké ", s_exp, flags=re.IGNORECASE)
    s_exp = re.sub(r"^(?:Dol|Dl)\.\s*", "Dolný ", s_exp, flags=re.IGNORECASE)
    s_exp = re.sub(r"^(?:Mal|Ml)\.\s*", "Malé ", s_exp, flags=re.IGNORECASE)

    # Handle 2-word inversion where adjective is second (e.g. `Čepeň Horný` -> `Horný Čepeň`, `Šúrovce Veľké` -> `Veľké Šúrovce`)
    parts = s_exp.split()
    if len(parts) == 2:
        p1_f = fold_text_diacritics(parts[1]).rstrip(".")
        if p1_f in ("horny", "horni", "horne", "horna", "hor", "hr"):
            s_exp = f"Horný {parts[0]}"
        elif p1_f in ("dolny", "dolni", "dolne", "dolna", "dol"):
            s_exp = f"Dolný {parts[0]}"
        elif p1_f in ("velky", "velke", "velka", "vel", "vell"):
            s_exp = f"Veľké {parts[0]}"
        elif p1_f in ("maly", "male", "mala", "mal"):
            s_exp = f"Malé {parts[0]}"

    return s_exp


def normalize_census_municipality(
    raw_muni: str | None,
    sheet_municipality: str | None = None,
    sheet_district: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Normalizes census `a) Obec` (birthplace municipality) against:
    1. Abbreviation & inversion expander (`Hor. Čepeň`, `Jepen Horni`, `Vel. Suvorce`, `Valtasin`, ...)
    2. `REGIONAL_GAZETTEER_100KM` + `data/regional_places_cache.json` + `VILLAGE_MAP` + `slovakiana_facets_all.json`
    Returns `(canonical_municipality, inferred_district, inferred_country)`.
    """
    if not raw_muni:
        return None, None, None
    cleaned = _strip_place_rubric_noise(raw_muni)
    if not cleaned or cleaned in ("/", "-", "//", '"'):
        return None, None, None

    f_raw = fold_text_diacritics(cleaned)
    f_tokens = re.sub(r"[^a-z]+", " ", f_raw).strip()
    if not f_tokens or f_tokens in _NON_PLACE_GARBAGE_TOKENS:
        return None, None, None
    # Reject obvious marital/sex/role OCR garbage tokens
    if any(f_tokens == g or f_tokens.startswith(g + " ") for g in _NON_PLACE_GARBAGE_TOKENS):
        return None, None, None

    expanded = _expand_place_abbreviations_and_inversions(cleaned)
    if expanded in REGIONAL_GAZETTEER_100KM:
        d_inf, c_inf = REGIONAL_GAZETTEER_100KM[expanded]
        return expanded, (d_inf or None), (c_inf or None)

    v_sk, _, _ = normalize_village(expanded)
    if v_sk and v_sk in REGIONAL_GAZETTEER_100KM:
        d_inf, c_inf = REGIONAL_GAZETTEER_100KM[v_sk]
        return v_sk, (d_inf or None), (c_inf or None)

    exp_f = fold_text_diacritics(expanded)
    gazetteer = get_regional_municipality_gazetteer(sheet_municipality, sheet_district)
    cache_places = (_load_regional_places_cache().get("places") or {})

    sheet_mf = fold_text_diacritics(sheet_municipality) if sheet_municipality else ""
    best_muni: str | None = None
    best_score = 0.0

    for cand in gazetteer:
        cf = fold_text_diacritics(cand)
        if cf == exp_f:
            info = REGIONAL_GAZETTEER_100KM.get(cand)
            if info:
                return cand, (info[0] or None), (info[1] or None)
            c_info = cache_places.get(cf)
            if isinstance(c_info, dict):
                return cand, (c_info.get("district") or None), (c_info.get("country") or "Slovensko")
            return cand, None, "Slovensko"

        dist_cost = _paleographic_distance(exp_f, cf)
        max_l = max(len(exp_f), len(cf), 1)
        sim = max(0.0, 1.0 - (dist_cost / max_l))
        seq = difflib.SequenceMatcher(None, exp_f, cf).ratio()
        score = max(sim, seq)
        if sheet_mf and cf == sheet_mf:
            score += 0.10
        if score > best_score:
            best_score = score
            best_muni = cand

    if best_muni and best_score >= 0.72:
        info = REGIONAL_GAZETTEER_100KM.get(best_muni)
        if info:
            return best_muni, (info[0] or None), (info[1] or None)
        c_info = cache_places.get(fold_text_diacritics(best_muni))
        if isinstance(c_info, dict):
            return best_muni, (c_info.get("district") or None), (c_info.get("country") or "Slovensko")
        return best_muni, None, "Slovensko"

    # Also check full Slovakiana 3,712-municipality list for exact or very close match
    _ensure_slovakiana_facets_loaded()
    for sm in _SLOVAKIANA_MUNICIPALITIES:
        sm_f = fold_text_diacritics(sm)
        if sm_f == exp_f:
            return sm, None, "Slovensko"

    if v_sk:
        return v_sk, None, "Slovensko"
    return expanded, None, None


def normalize_census_birthplace_fields(
    municipality_raw: str | None,
    district_raw: str | None,
    country_raw: str | None,
    *,
    sheet_municipality: str | None = None,
    sheet_district: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Jointly normalizes `(birth_place_municipality, birth_place_district, birth_place_country)`
    for a 1930/1940 census row using the ~100km Regional Gazetteer, FamilySearch Places cache,
    and Slovakiana municipality/district facets.
    """
    canon_muni, inferred_dist, inferred_cnt = normalize_census_municipality(
        municipality_raw,
        sheet_municipality=sheet_municipality,
        sheet_district=sheet_district,
    )
    canon_dist = normalize_census_district(
        district_raw,
        sheet_district=sheet_district,
        inferred_district=inferred_dist,
    )
    # If municipality has a known canonical district in REGIONAL_GAZETTEER_100KM, prefer its authoritative district & country
    if canon_muni and canon_muni in REGIONAL_GAZETTEER_100KM:
        known_d, known_c = REGIONAL_GAZETTEER_100KM[canon_muni]
        if known_d:
            canon_dist = known_d
        elif not known_d and known_c != "Slovensko":
            canon_dist = None
        if known_c:
            inferred_cnt = known_c

    canon_cnt = normalize_census_country(
        country_raw,
        inferred_country=inferred_cnt or ("Slovensko" if canon_muni else None),
    )
    if canon_muni and canon_muni in REGIONAL_GAZETTEER_100KM:
        _, known_c = REGIONAL_GAZETTEER_100KM[canon_muni]
        if known_c and known_c != "Slovensko":
            canon_cnt = known_c

    return canon_muni, canon_dist, canon_cnt

