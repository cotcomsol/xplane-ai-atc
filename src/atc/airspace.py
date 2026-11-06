"""US Class B and C primary airports.

Generated from the FAA Class_Airspace dataset (ADDS open data, ArcGIS service
services6.arcgis.com/ssFJjBXIUyZDrSYZ/.../Class_Airspace), fetched 2026-11-05.
The dataset names one airport per Class B area, so the other primary airports of
multi-airport Class B areas are added by hand (CLASS_B_EXTRA).
Airports with a tower that aren't listed are treated as Class D.
"""

CLASS_B_EXTRA = frozenset({"KBWI", "KDAL", "KEWR", "KHOU", "KIAD", "KLGA", "KNKX"})

CLASS_B = frozenset({
    "KATL", "KBOS", "KCLE", "KCLT", "KCVG", "KDCA", "KDEN", "KDFW", "KDTW", "KIAH", "KJFK", "KLAS",
    "KLAX", "KMCI", "KMCO", "KMEM", "KMIA", "KMSP", "KMSY", "KORD", "KPHL", "KPHX", "KPIT", "KSAN",
    "KSEA", "KSFO", "KSLC", "KSTL", "KTPA", "PHNL",
}) | CLASS_B_EXTRA

CLASS_C = frozenset({
    "KABE", "KABI", "KABQ", "KACY", "KALB", "KAMA", "KAUS", "KAVL", "KBAB", "KBAD", "KBDL", "KBGR",
    "KBHM", "KBIL", "KBNA", "KBOI", "KBTR", "KBTV", "KBUF", "KBUR", "KCAE", "KCAK", "KCBM", "KCHA",
    "KCHS", "KCID", "KCMH", "KCMI", "KCOS", "KCRP", "KCRW", "KDAB", "KDAY", "KDJT", "KDLF", "KDMA",
    "KDSM", "KDYS", "KELP", "KEVV", "KFAT", "KFAY", "KFLL", "KFNT", "KFWA", "KGEG", "KGRB", "KGRR",
    "KGSO", "KGSP", "KHRL", "KHSV", "KICT", "KIND", "KISP", "KJAN", "KJAX", "KLAN", "KLBB", "KLEX",
    "KLFT", "KLIT", "KLNK", "KMAF", "KMDT", "KMDW", "KMHT", "KMKE", "KMLI", "KMOB", "KMRY", "KMSN",
    "KMYR", "KNDZ", "KNPA", "KNUW", "KOAK", "KOFF", "KOKC", "KOMA", "KONT", "KORF", "KPDX", "KPIA",
    "KPNS", "KPOB", "KPVD", "KPWM", "KRDU", "KRIC", "KRIV", "KRNO", "KROA", "KROC", "KRSW", "KSAT",
    "KSAV", "KSBA", "KSBN", "KSDF", "KSFB", "KSGF", "KSHV", "KSJC", "KSKA", "KSMF", "KSNA", "KSPI",
    "KSRQ", "KSSC", "KSYR", "KTIK", "KTLH", "KTOL", "KTUL", "KTUS", "KTYS", "KXNA", "PANC", "PHOG",
})


def airspace_class(airport: dict) -> str:
    """'B', 'C', 'D' (towered) or 'G/E' (no tower), for the airport's surface area."""
    ident = airport["id"].upper()
    if ident in CLASS_B:
        return "B"
    if ident in CLASS_C:
        return "C"
    return "D" if any(f["type"] == "TOWER" for f in airport.get("freqs", [])) else "G/E"
