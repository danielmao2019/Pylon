goal: initialize code skeleton docs and restructure the model-layer mesh texture extraction module.

guidelines:
1. the new model-layer skeletons live under `./docs/code_structure/models/mesh/`, split into `code_structure.md`, `folder_structure.md`, and `tests_structure.md`.
2. references:
   1. existing `./docs/code_structure/data/structures/three_d/mesh/`.
   2. the code skeleton related work in claude session transcripts, claude memories, and task specs, of any of the following projects:
      1. Pylon, Pylon-*
      2. iVISION-PCR, iVISION-PCR-*
      3. OfficeHours, OfficeHours-*
      4. sthetic-face, sthetic-face-*
3. task scope: the deliverable lives in the model layer, but data-layer touching is NOT off-limits. data-layer code changes (`data/structures/three_d/mesh/`) are in-scope whenever they are a hard prerequisite for the model-layer restructure, and every such data-layer code change must come with the matching back-propagated update to the data-layer skeleton at `./docs/code_structure/data/structures/three_d/mesh/`.
