"""Auto-register all atoms into the global KB."""
from knowledge.atoms.mathematician_atoms import *
from knowledge.atoms.cryptographer_atoms import *
from knowledge.atoms.physicist_atoms import *
from knowledge.atoms.linguist_atoms import *
from knowledge.atoms.systems_engineer_atoms import *
from knowledge.atoms.shared_atoms import *
from knowledge.atoms.crypto_atoms import *
from knowledge.registry import KB
print(f"Knowledge Registry: {len(KB)} atoms loaded")
