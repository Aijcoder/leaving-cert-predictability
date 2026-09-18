"""Phase 4 (PHY): codebook PHY_v1 from the 1999 Higher Level syllabus headings (§11.1).

Topics group numbered syllabus headings (outline indexes from phy_outline.py). `includes` are phrases from the
headings' "Depth of Treatment" text. Designed without looking at topic frequencies (§11.1).
Run: python -m lcbank.codebook.build_phy
"""
import csv
import hashlib
import json

from lcbank.codebook.common import header, phrases, write_codebook
from lcbank.common.paths import DATA_DERIVED, MANIFEST

# topic_id, L1 (syllabus section), name, outline indexes, excludes, boundary rules
TOPICS = [
    ("PHY.MEC.01", "Mechanics", "Linear motion, vectors and scalars", [0, 1],
     ["Motion in a circle (PHY.MEC.03)", "Projectile-free SHM oscillations (PHY.MEC.07)"],
     ["Equations of motion, velocity-time graphs and vector resolution belong here unless the context is circular motion, gravitation or SHM."]),
    ("PHY.MEC.02", "Mechanics", "Newton's laws of motion and conservation of momentum", [2, 3],
     ["Centripetal force (PHY.MEC.03)", "Momentum conservation in nuclear/particle reactions (PHY.PAR.01)"],
     ["F = ma, momentum, impulse and collisions/explosions belong here."]),
    ("PHY.MEC.03", "Mechanics", "Circular motion", [4], ["Satellite orbits using gravitation (PHY.MEC.04)"],
     ["Angular velocity, centripetal acceleration and force in a general circle belong here."]),
    ("PHY.MEC.04", "Mechanics", "Gravitation", [5], [],
     ["Newton's law of universal gravitation, weight, g variation and satellites/orbital period belong here, including their circular-motion algebra."]),
    ("PHY.MEC.05", "Mechanics", "Density and pressure", [6], [], ["Boyle's law and Archimedes/flotation belong here."]),
    ("PHY.MEC.06", "Mechanics", "Moments and conditions for equilibrium", [7, 8], [], []),
    ("PHY.MEC.07", "Mechanics", "Simple harmonic motion and Hooke's law", [9], ["Stationary waves (PHY.SND.02)"],
     ["The simple pendulum (including measuring g with a pendulum) belongs here."]),
    ("PHY.MEC.08", "Mechanics", "Work, energy and power", [10, 11, 12], ["Electrical power in circuits (PHY.ELE.06)"],
     ["Mechanical energy conservation (Ep = mgh, Ek = mv²/2), efficiency and power belong here."]),
    ("PHY.TMP.01", "Temperature", "Temperature and thermometers", [15, 16, 17], [],
     ["Thermometric properties and calibration curves belong here."]),
    ("PHY.HEA.01", "Heat", "Heat capacity and latent heat", [18, 19, 20], ["Heat transfer mechanisms (PHY.HEA.02)"],
     ["Calorimetry calculations and specific heat / latent heat experiments belong here."]),
    ("PHY.HEA.02", "Heat", "Heat transfer: conduction, convection and radiation", [21, 22, 23], [],
     ["U-values, insulation and the solar constant belong here."]),
    ("PHY.WAV.01", "Waves", "Wave properties and wave phenomena", [24, 25],
     ["Sound-specific content (PHY.SND.01)", "Light-specific diffraction/interference (PHY.LIG.04)"],
     ["General wave motion (c = fλ, transverse/longitudinal, stationary waves in general) belongs here when not tied to sound or light."]),
    ("PHY.WAV.02", "Waves", "Doppler effect", [26], [], ["All Doppler calculations and applications (sound, light, red shift) belong here."]),
    ("PHY.SND.01", "Vibrations and sound", "Sound: nature, characteristics, resonance and intensity level", [27, 28, 29, 31],
     ["Frequencies of strings and pipes (PHY.SND.02)"],
     ["Speed of sound, loudness/pitch/quality, resonance, sound intensity and decibels belong here."]),
    ("PHY.SND.02", "Vibrations and sound", "Vibrations in strings and pipes", [30], [],
     ["Laws of stretched strings, harmonics in pipes and related experiments belong here."]),
    ("PHY.LIG.01", "Light", "Reflection and mirrors", [32, 33], [], []),
    ("PHY.LIG.02", "Light", "Refraction and total internal reflection", [34, 35], ["Lenses (PHY.LIG.03)"],
     ["Snell's law, refractive index, critical angle and optical fibres belong here."]),
    ("PHY.LIG.03", "Light", "Lenses", [36], [], ["Lens formula, power of a lens, the eye and lens combinations belong here."]),
    ("PHY.LIG.04", "Light", "Diffraction, interference and polarisation of light", [37, 38], [],
     ["Diffraction grating (nλ = d sinθ), Young's slits and polarisation belong here."]),
    ("PHY.LIG.05", "Light", "Dispersion, colour, electromagnetic spectrum and the spectrometer", [39, 40, 41, 42], [], []),
    ("PHY.ELE.01", "Electricity", "Static electricity: charges and electroscope", [44, 45, 46, 47],
     ["Coulomb's law and fields (PHY.ELE.02)"], []),
    ("PHY.ELE.02", "Electricity", "Coulomb's law, electric field and potential difference", [48, 49, 50], [], []),
    ("PHY.ELE.03", "Electricity", "Capacitance", [51], [], ["Capacitor energy and parallel-plate capacitance belong here."]),
    ("PHY.ELE.04", "Electricity", "Electric current, emf and conduction in materials", [52, 53, 54],
     ["Diode applications, transistor, logic gates (PHY.APE.01)"],
     ["Conduction in metals, solutions, gases and semiconductors (intrinsic/extrinsic, doping, p-n junction bias) belongs here."]),
    ("PHY.ELE.05", "Electricity", "Resistance, resistivity and potential divider", [55, 56], [],
     ["Ohm's law, series/parallel resistance, Wheatstone bridge, metre bridge and thermistor/LDR behaviour belong here."]),
    ("PHY.ELE.06", "Electricity", "Effects of electric current and domestic circuits", [57, 58], [],
     ["Joule's law (heating), electrical power P = VI, kWh, fuses, MCBs, RCDs and bonding belong here."]),
    ("PHY.ELE.07", "Electricity", "Magnetism and magnetic fields", [59, 60, 82], [],
     ["Field patterns, solenoids and the electromagnetic relay (Option 2 item 1) belong here."]),
    ("PHY.ELE.08", "Electricity", "Current and moving charge in a magnetic field", [61, 83], [],
     ["F = BIL, F = qvB, magnetic flux density, the d.c. motor, moving-coil meters and loudspeaker (Option 2 item 2) belong here."]),
    ("PHY.ELE.09", "Electricity", "Electromagnetic induction, alternating current and mutual/self-induction", [62, 63, 64, 84, 85], [],
     ["Faraday/Lenz, generators, rms values, transformers and induction coil (Option 2 items 3-4) belong here."]),
    ("PHY.MOD.01", "Modern physics", "The electron and thermionic emission", [65, 66], ["X-rays (PHY.MOD.03)"],
     ["Cathode ray tube, e/m and the CRO belong here."]),
    ("PHY.MOD.02", "Modern physics", "Photoelectric emission and the photon", [67], [], ["E = hf, work function and photocells belong here."]),
    ("PHY.MOD.03", "Modern physics", "X-rays", [68], [], []),
    ("PHY.MOD.04", "Modern physics", "Structure of the atom and nucleus", [69, 70], ["Radioactive decay (PHY.MOD.05)"],
     ["Rutherford scattering, Bohr model, energy levels and line spectra, nuclear notation and isotopes belong here."]),
    ("PHY.MOD.05", "Modern physics", "Radioactivity and ionising radiation", [71, 73], ["Fission and fusion (PHY.MOD.06)"],
     ["Alpha/beta/gamma properties, detectors, half-life and decay law, health hazards belong here."]),
    ("PHY.MOD.06", "Modern physics", "Nuclear energy: fission and fusion", [72], ["Particle accelerator reactions (PHY.PAR.01)"],
     ["E = mc² for fission/fusion energy, chain reactions and reactors belong here."]),
    ("PHY.PAR.01", "Option 1: Particle physics", "Accelerators, conservation laws and mass-energy conversion in reactions", [74, 75, 76, 77], [],
     ["Cockcroft-Walton, circular accelerators, neutrino prediction from momentum conservation and pair production energy balance belong here."]),
    ("PHY.PAR.02", "Option 1: Particle physics", "Fundamental forces, particle families, antimatter and quarks", [78, 79, 80, 81], [], []),
    ("PHY.APE.01", "Option 2: Applied electricity", "Diode applications, the transistor and logic gates", [86, 87, 88],
     ["p-n junction physics (PHY.ELE.04)"], ["Rectification, LEDs, photodiodes, transistor amplifiers/switches and AND/OR/NOT gates belong here."]),
]

GLOBAL_RULES = [
    "Label each scoring step by the physics it assesses, not by where the question sits on the paper.",
    "Mandatory experiment questions (Section A) are coded by the topic of the quantity measured or law verified; apparatus, procedure, precautions, graph drawing and calculation steps within the experiment take the same topic.",
    "A definition, unit or symbol is coded to the topic where that quantity is defined in the syllabus.",
    "Option 2 (Applied electricity) items on solenoids/relays, motors/loudspeakers, induction coils and a.c. generators are coded to the core Electricity topics PHY.ELE.07-09; only diode applications, transistors and logic gates use PHY.APE.01.",
    "Everyday applications and STS content (e.g. 'give one use of…') are coded to the topic of the underlying physics.",
    "A reading-comprehension question is coded step by step according to the physics each step assesses.",
    "Use UNCLEAR only when a step cannot be assigned to a single topic with reasonable confidence.",
]


def build():
    outline = json.loads((DATA_DERIVED / "codebooks" / "sources" / "PHY_syllabus_outline.json").read_text())
    used = sorted(i for t in TOPICS for i in t[3])
    content_idx = [i for i, o in enumerate(outline) if o["depth"] and len(o["heading"]) < 45 and i not in (13, 14, 43)]
    missing = sorted(set(content_idx) - set(used))
    doubled = sorted({i for i in used if used.count(i) > 1})
    if missing or doubled:
        raise ValueError(f"syllabus headings not covered exactly once: missing {missing}, doubled {doubled}")
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        syl = next(r for r in csv.DictReader(f) if r["doc_id"] == "PHY-syllabus-1999")
    outline_sha = hashlib.sha256((DATA_DERIVED / "codebooks" / "sources" / "PHY_syllabus_outline.json").read_bytes()).hexdigest()
    cb = header("PHY", "v1", [{"doc_id": syl["doc_id"], "url": syl["url"], "sha256": syl["sha256"],
                               "pages": "29-48 (Higher Level syllabus content)", "outline_sha256": outline_sha}],
                GLOBAL_RULES)
    topics = []
    for tid, l1, name, idxs, excludes, rules in TOPICS:
        heads = [outline[i] for i in idxs]
        ref = "; ".join(f"{h['section'].title()} > {(h['subsection'] or '').title()} > {h['heading']} (p{h['page']})" for h in heads)
        includes = []
        for h in heads:
            includes += phrases(h["depth"], max_n=5)
        topics.append({"topic_id": tid, "L1": l1, "name": name, "syllabus_ref": ref, "includes": includes[:10],
                       "excludes": excludes, "boundary_rules": rules, "examples": []})
    cb["topics"] = topics
    return cb


if __name__ == "__main__":
    cb = build()
    path, sha = write_codebook(cb)
    print(path, len(cb["topics"]), "topics", "sha256", sha)
