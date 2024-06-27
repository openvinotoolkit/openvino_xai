# Change Log

## [1.0.0]

### Summary

* Support generation of classification and detection paj-class and per-image saliency maps
* Enable white-box (ReciproCAM) and black-box (RISE) eXplainable AI algorithms
* Support CNN and transformer-based architectures (validation on diverse set of timm models)
* Enable Explainer (stateful object) as the main interface for XAI algorithms
* Expose `insert_xai` functional API to support XAI head insertion for OpenVINO IR models

### What's Changed

* Support classification task by @negvet in https://github.com/intel-sandbox/openvino_xai/commit/dd5fd9b73fe8c12e2d741792043372bcd900a850
* Support detection task by @negvet in https://github.com/intel-sandbox/openvino_xai/commit/84f285f2f40a8b1fc50a8cd49798aae37afd58dc
* Support Model API as inference engine by @negvet in https://github.com/intel-sandbox/openvino_xai/commit/5f575f122dedc0461975bd58f81e730a901a69a6
* Support classification validation on OTX models by @negvet in https://github.com/intel-sandbox/openvino_xai/commit/008f25f3b09e480906d8edcf043da2ece9ca3e6b
* Update requirements, add --output parameter for run.py by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/1
* Separate cls and det exampels and improve cls tests by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/2
* Enable scaling and postprocessing. Add object for map by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/3
* Fix saliency map dtype by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/5
* Minor fix for map dtype by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/6
* Support just IR update by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/7
* Check for xai before inserting by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/8
* Support parameter objects by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/9
* Update setup script by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/10
* Support benchmark test for timm==0.9.5 models (white-box) by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/11
* API docs by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/12
* Add model scope to README.md by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/13
* Add RISE BlackBox explaining algorithm for classification by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/15
* Update README.md by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/17
* Refactor: split explain file onto different algorithms by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/16
* Update BB and examples by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/18
* Fix RISE memory issue + minor updates by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/20
* Optimize RISE by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/22
* Update scope for BB by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/24
* Refactor saliency maps by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/21
* Fix labels for saved saliency maps by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/25
* Fix bugs and update test coverage by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/26
* Support benchmark test for timm==0.9.5 models (black-box) by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/23
* Update image name by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/27
* Update interpolation in black_box.py by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/29
* Update insertion point search by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/31
* Add getting started by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/28
* API update and refactor by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/32
* Fix and refactor tests by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/33
* Align Detection White Box algo with OTX by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/34
* Add unit and integration tests for WB detection by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/38
* Enable VITReciprocam by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/35
* Fix licenses, update deps by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/40
* Fix reference values after color channel order update by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/41
* Use ovc instead of mo by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/43
* API docs update by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/45
* Add XAI Classification notebook by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/48
* Add Saliency map interpretation notebook by @GalyaZalesskaya in https://github.com/intel-sandbox/openvino_xai/pull/49
* Fix tests by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/50
* Use pyproject.toml instead of setup.py by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/52
* Fix setup config by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/53
* Support preprocess_fn & postprocess_fn by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/51
* Support target_explain_labels of different types by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/54
* Make MAPI an optional dep by @sovrasov in https://github.com/intel-sandbox/openvino_xai/pull/55
* Remove MAPI and fix tests by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/56
* Docs update by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/57
* Refactor postprocessing by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/58
* Fix mutable default value in `ExplanationParameters` by `default_factory` by @goodsong81 in https://github.com/intel-sandbox/openvino_xai/pull/59
* Update pre-commit + code style by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/60
* Update types by @negvet in https://github.com/intel-sandbox/openvino_xai/pull/62
* Add unit test coverage setting by @goodsong81 in https://github.com/intel-sandbox/openvino_xai/pull/63
* Add LICENSE and SECURITY.md by @goodsong81 in https://github.com/intel-sandbox/openvino_xai/pull/64
* Add CHANGLOG.md by @goodsong81 in https://github.com/intel-sandbox/openvino_xai/pull/65
* Update project title and repo links by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/1
* Add pre-merge GitHub Actions workflow by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/2
* Add security code scan workflow by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/4
* Add daily E2E test workflow by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/6
* Refactor names/folders/objects for better verbosity by @GalyaZalesskaya in https://github.com/openvinotoolkit/openvino_xai/pull/5
* Fix daily workflow by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/8
* Support XAI Method Abstract Interface by @negvet in https://github.com/openvinotoolkit/openvino_xai/pull/7
* Update version to 1.0.0rc0 by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/10
* Move insert_xai into separate functional api module by @negvet in https://github.com/openvinotoolkit/openvino_xai/pull/11
* Add fuzzing test by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/13
* Support CodeQL for security code scan by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/14
* Route e2e test to "large-disk" labeled ci-runner by @yunchu in https://github.com/openvinotoolkit/openvino_xai/pull/17
* Flatten parameter objects by @negvet in https://github.com/openvinotoolkit/openvino_xai/pull/16
* Add mount option for ci script by @yunchu in https://github.com/openvinotoolkit/openvino_xai/pull/21
* Remove target explain group by @negvet in https://github.com/openvinotoolkit/openvino_xai/pull/25
* Add third-party-programs.txt by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/27
* Add full timm functional tests by @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/20
* Add performance tests @goodsong81 in https://github.com/openvinotoolkit/openvino_xai/pull/31

### New Contributors

* @negvet made their first contribution in https://github.com/intel-sandbox/openvino_xai/commit/dd5fd9b73fe8c12e2d741792043372bcd900a850
* @GalyaZalesskaya made their first contribution in https://github.com/intel-sandbox/openvino_xai/pull/1
* @sovrasov made their first contribution in https://github.com/intel-sandbox/openvino_xai/pull/10
* @goodsong81 made their first contribution in https://github.com/intel-sandbox/openvino_xai/pull/59
* @yunchu made their first contribution in https://github.com/openvinotoolkit/openvino_xai/pull/17
