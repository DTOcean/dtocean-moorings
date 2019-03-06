# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.0.0] - 2019-03-06

### Added

- Raise a RuntimeError if the calculated device draft differs from the given 
  equilibrium draft by more than 5%.
- Return human readable error if anchors are not applicable to a soil type.

### Changed

- Split core module into 5 separate modules, one for each class.
- Broke down Loads.gpnearloc into smaller functions and fixed various bugs.
- Fix use of gpnear coordinate for finding seabed slope in Foundations class.
- Catch slope angle exceeding friction angle for gravity or shallow
  foundations.
- Improved calculation efficiency by reducing number of numpy function calls.
- Changed function of prefound argument to define a preferred foundation type
  rather than a single type. If the preferred option can not be found then the
  cheapest alternative is used.

### Removed

- Removed non-working islay and shetland example files and data.

### Fixed

- Fixed issues with inheritance in Main and Umb classes.
- Fixed issue with groutpilebondstr attribute not being initialised in Found
  class.
- Ensured drag anchor database ids are added to the bill of materials, rather
  than just "drag".
- Fixed bug in vertical distance requirements for moorings lines.
- Fixed bug in device draft calculation.
- Fixed issue where the global system position was incorrectly using positive
  depth.
- Fixed functionality for using a single design for all foundations of a device
  ("uniary").
- Fixed bug with calculation of umbilical top connection point.
- Fixed bug in device position retuned by the mooreqav method.
- Fixed ULS and ALS displacement testing logic.
- Fixed bug with non-float arguments to numpy.linalg.lstsq function.

## [1.0.0] - 2017-01-05

### Added

- Initial import of dtocean-gui from SETIS.


