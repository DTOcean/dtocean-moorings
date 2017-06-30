[![appveyor](https://ci.appveyor.com/api/projects/status/github/DTOcean/dtocean-moorings?branch=master&svg=true)](https://ci.appveyor.com/project/DTOcean/dtocean-moorings)
[![codecov](https://codecov.io/gh/DTOcean/dtocean-moorings/branch/master/graph/badge.svg)](https://codecov.io/gh/DTOcean/dtocean-moorings)
[![Lintly](https://lintly.com/gh/DTOcean/dtocean-moorings/badge.svg)](https://lintly.com/gh/DTOcean/dtocean-moorings/)
[![release](https://img.shields.io/github/release/DTOcean/dtocean-moorings.svg)](https://github.com/DTOcean/dtocean-moorings/releases/latest)

## Component database keys and values

* item1: Mooring or foundation system
* item2: Name
* item3: Subname
* item4: [Material, Grade, Colour]
* item5: Strength and mechanical properties. For piles [yield stress, Young's modulus]. For ropes [minimum break load, [load, axial stiffness]]. For cables [minimum break load, minimum bend radius]. All other components: [minimum break load, axial stiffness]
* item6: Size. For piles/suction caissons [diameter, thickness]. For anchors [width, depth, height, connecting size]. For chains, forerunner assemblies, shackles and swivels [diameter, connecting length]. For ropes and cables [outer diameter]
* item7: Mass. For piles/suction caissons, chains, ropes, cables and forerunner assemblies [dry mass per unit length, wet mass per unit length]. For anchors, shackles and swivels [dry unit mass, wet unit mass]
* item8: Environmental impact
* item9: Anchor coefficients
* item10: Failure rates
* item11: Cost
