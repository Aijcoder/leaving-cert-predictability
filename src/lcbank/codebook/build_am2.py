"""Phase 4 (AM2): codebook AM2_v1 from the Leaving Certificate Applied Mathematics specification (2021),
strands and learning outcomes on pages 16-21. `includes` quote the specification's "students learn about" /
"students should be able to" wording. Designed without looking at topic frequencies (§11.1).
Run: python -m lcbank.codebook.build_am2
"""
import csv

from lcbank.codebook.common import header, write_codebook
from lcbank.common.paths import MANIFEST

S1, S2 = "Strand 1: Mathematical modelling", "Strand 2: Mathematical modelling with networks and graphs"
S3, S4 = "Strand 3: Mathematically modelling the physical world; kinematics and dynamics", \
    "Strand 4: Mathematically modelling a changing world"

TOPICS = [
    ("AM2.MOD.01", S1, "The mathematical modelling process", "p16: problem-solving cycle; formulating problems; translating problems into mathematics; computing solutions; evaluating solutions",
     ["determine what assumptions are necessary to simplify the problem situation", "translate the information given in the problem together with the assumptions into a mathematical model",
      "interpret the mathematical solution in terms of the original situation", "refine a model and use it to predict a better solution to the problem; iterate the process"],
     [], ["Steps that ask for assumptions, limitations or refinement of a model belong here, whatever the strand of the model."]),
    ("AM2.NET.01", S2, "Networks, graph terminology and representation", "p17: networks and their associated terminology",
     ["represent real-world situations in the form of a network", "vertex / node, edge/arc, weight, path, cycle",
      "distinguish between connected and disconnected graphs, and between directed and undirected graphs"],
     ["Adjacency matrix algebra (AM2.NET.02)"], []),
    ("AM2.NET.02", S2, "Matrices, matrix algebra and adjacency matrices", "p17: matrices, matrix algebra and adjacency",
     ["represent a graph using an adjacency matrix, and reconstruct a graph from its adjacency matrix",
      "perform multiplication of square matrices", "interpret the product of adjacency matrices"], [], []),
    ("AM2.NET.03", S2, "Minimum spanning trees (Kruskal, Prim)", "p17: minimum spanning trees",
     ["concepts of tree, spanning tree, minimum spanning tree", "use appropriate algorithms to find minimum spanning trees"], [], []),
    ("AM2.NET.04", S2, "Shortest paths: Dijkstra's algorithm", "p17: algorithms (Dijkstra)",
     ["apply Dijkstra's algorithm to find the shortest paths in a weighted undirected and directed network"],
     ["Bellman / dynamic programming (AM2.NET.05)"], []),
    ("AM2.NET.05", S2, "Dynamic programming and Bellman's principle of optimality", "p17: dynamic programming and shortest paths in multi-stage problems",
     ["use and apply dynamic programming terminology, such as stage, state, optimal policy",
      "apply Bellman's Principle of Optimality to find the shortest paths in a weighted directed acyclic network",
      "stock control, routing problems, allocation of resources, equipment replacement and maintenance"], [], []),
    ("AM2.NET.06", S2, "Evaluating algorithms: greedy versus dynamic programming", "p17: algorithms",
     ["distinguish between those algorithms which are greedy and those which use dynamic programming",
      "justify the use of algorithms in terms of correctness and their ability to yield an optimal solution",
      "evaluate different techniques for solving shortest-route problems"], [],
     ["Running an algorithm is coded to the algorithm's topic (AM2.NET.03-05); comparing or justifying algorithms belongs here."]),
    ("AM2.NET.07", S2, "Project scheduling and critical path analysis", "p17: analysis of project scheduling networks",
     ["apply network concepts to project scheduling", "critical path, early times, late times and floats"], [], []),
    ("AM2.KIN.01", S3, "Kinematics in one dimension with constant acceleration", "p18: kinematics: particle motion in one direction",
     ["position, displacement, velocity, acceleration and time", "displacement-time graphs, velocity-time graphs",
      "v = u + at; s = ut + ½at²; v² = u² + 2as"], ["Variable acceleration using calculus (AM2.KIN.02)"], []),
    ("AM2.KIN.02", S3, "Kinematics with calculus: variable acceleration", "p18: velocity and acceleration as rates of change",
     ["interpret velocity and acceleration as derivatives",
      "transform the function describing one quantity (displacement, velocity, acceleration) into functions describing the other two",
      "derive the kinematic formulae of motion using calculus"], ["Motion under drag forces (AM2.DYN.04)"], []),
    ("AM2.KIN.03", S3, "Vectors and particle motion in two dimensions", "p19: particle motion in 2D; elementary vector algebra and calculus",
     ["represent vectors in terms of components along unit vectors in 2 fixed orthogonal directions and in polar form",
      "calculate and interpret the dot product of vectors", "elementary vector calculus"], ["Projectiles (AM2.KIN.04)"],
     ["Relative-motion and vector-velocity problems without projectile motion belong here."]),
    ("AM2.KIN.04", S3, "Projectile motion", "p19: projectile motion",
     ["time of flight, maximum height, maximum range, horizontal planes", "solve constant acceleration projectile motion problems"], [], []),
    ("AM2.DYN.01", S3, "Forces, Newton's laws and motion on planes", "p19: forces acting on a particle",
     ["draw free-body force diagrams for a particle on a smooth rigid fixed horizontal or inclined plane",
      "resolve forces on rough and smooth surfaces", "applied, normal reaction, frictional, resistant, tension, gravitational forces; coefficient of friction",
      "solve dynamic problems involving the motion of a particle under a constant resultant force"],
     ["Connected masses (AM2.DYN.02)", "Drag (AM2.DYN.04)"], []),
    ("AM2.DYN.02", S3, "Connected masses", "p19: connected masses", ["solve dynamic problems involving connected masses"], [],
     ["Pulley and string systems with two or more particles belong here, including their friction."]),
    ("AM2.DYN.03", S3, "Momentum, impulse and collisions", "p19: momentum, impulse, conservation of momentum, collisions",
     ["conservation of momentum for a two-particle system in one and two dimensions", "Newton's experimental laws for collisions; coefficient of restitution",
      "solve dynamic problems involving particles that collide directly and obliquely"], [], []),
    ("AM2.DYN.04", S3, "Resistive forces and drag", "p19: drag: liquid, aerodynamic where F ∝ vⁿ",
     ["solve dynamic problems involving resistive forces that are proportional to vⁿ", "integration by parts / by substitution applied to motion"],
     [], ["Setting up and integrating F = m dv/dt with a velocity-dependent resistance belongs here (not AM2.DEQ.*)."]),
    ("AM2.DYN.05", S3, "Work, energy and conservation of energy", "p20: work, energy, conservative forces, conservation of energy",
     ["define work done", "gravitational potential energy and kinetic energy and how they relate to work done",
      "conservation of energy for variable conservative forces", "force exerted by a linear stretched or compressed elastic spring and stretched strings"],
     ["Energy in circular motion problems (AM2.DYN.06)"], []),
    ("AM2.DYN.06", S3, "Circular motion of a particle", "p20: circular motion of a particle",
     ["solve problems involving the dynamics of a particle moving in a horizontal or vertical circle"], [],
     ["Energy conservation used within a vertical-circle problem belongs here."]),
    ("AM2.DYN.07", S3, "Dimensional analysis", "p20: dimensional analysis",
     ["evaluate and articulate whether an answer is reasonable by analysing the dimensions"], [], []),
    ("AM2.DIF.01", S4, "Recurrence relations and modelling incremental change", "p21: recurrence relations; real-world phenomena involving incremental change",
     ["compute the first and higher differences of a given sequence of numbers", "derive difference equations for real-world phenomena involving incremental change",
      "Malthusian growth, restricted growth, interest/loan payment, supply and demand, spread of diseases"],
     ["Solving the difference equation (AM2.DIF.02)"], []),
    ("AM2.DIF.02", S4, "Solving difference equations", "p21: solving homogeneous and inhomogeneous difference equations",
     ["solve linear and non-linear difference equations", "analyse, interpret and solve difference equations in context"], [], []),
    ("AM2.DEQ.01", S4, "Formulating differential equations for continuous change", "p21: analysing real-world phenomena involving continuous change",
     ["identify real-world situations which can be suitably modelled by differential equations",
      "derive and interpret in context differential equations for real-world phenomena involving continuous change"],
     ["Solving (AM2.DEQ.02)", "Drag in dynamics (AM2.DYN.04)"], []),
    ("AM2.DEQ.02", S4, "Solving differential equations", "p21: techniques for solving differential equations",
     ["separation of variables", "first order separable; second order which can be reduced to first order",
      "numerical and graphical methods", "interpret the solution of differential equations in context"], [], []),
]

GLOBAL_RULES = [
    "Label each scoring step by the mathematics or model it assesses, not by the question number.",
    "Integration or differentiation techniques are coded to the model they serve (kinematics AM2.KIN.02, drag AM2.DYN.04, differential equations AM2.DEQ.*).",
    "Stating assumptions, limitations or refinements of a model → AM2.MOD.01.",
    "Running a named algorithm → that algorithm's topic; comparing or justifying algorithms → AM2.NET.06.",
    "Use UNCLEAR only when a step cannot be assigned to a single topic with reasonable confidence.",
]


def build():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        spec = next(r for r in csv.DictReader(f) if r["doc_id"] == "AM2-specification-2021-EN_1")
    cb = header("AM2", "v1", [{"doc_id": spec["doc_id"], "url": spec["url"], "sha256": spec["sha256"],
                               "pages": "16-21 (strands and learning outcomes)", "applies_to": ["AM2 2023-2026"]}],
                GLOBAL_RULES)
    cb["topics"] = [{"topic_id": tid, "L1": l1, "name": name, "syllabus_ref": ref, "includes": inc, "excludes": exc,
                     "boundary_rules": rules, "examples": []} for tid, l1, name, ref, inc, exc, rules in TOPICS]
    return cb


if __name__ == "__main__":
    cb = build()
    out, sha = write_codebook(cb)
    print(out, len(cb["topics"]), "topics", "sha256", sha)
