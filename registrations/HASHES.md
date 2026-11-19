# Original and released SHA256

Every analysis script and every registration text below was frozen during the
study. The released copy differs from the frozen copy only where a hard-coded
absolute path had to become an environment variable or a relative path, and
where a machine name or user name had to be removed. No seed, threshold, arm,
eligibility rule or decision rule was edited. `git log` of the private
repository holds the same statement.

The frozen originals are not part of this release; only their SHA256 is.

## Analysis scripts (`experiments/`)

| released file | original file | original SHA256 | released SHA256 | changed lines | note |
|---|---|---|---|---|---|
| `experiments/deletion_metrics/g5_dose.py` | `g5_dose.py` | `17f3324879caeebe8b176bb6c6249f037e7a11f039ba2b8fe11accd7b580a295` | `b5a6589826676b4fc29d390f48c8a3e77f42d4cd433defe7cb3bf7a43e565a4f` | 6 | path and anonymisation changes only |
| `experiments/deletion_metrics/merge_g5.py` | `merge_g5.py` | `70e6fd80e985b0207d5eb6e0b8c810dedfcc5329549e92035a11a051c511d7e6` | `5cd024013726ce0dcc3f8ac2569416a9b8022e33221ffd9805e259aaabf11057` | 2 | path and anonymisation changes only |
| `experiments/deletion_metrics/task_g5.py` | `task_g5.py` | `fd2bf6331d32e06c44c436be660dcc0bce343bdff30e9625c4a0d4f939268d9d` | `c57400d9b0df09314869f96638e2a808d1c45737aafce19e328702d2e5176e81` | 27 | path and anonymisation changes only |
| `experiments/deletion_metrics/task_g5_rt.py` | `task_g5_rt.py` | `62a5d003b0cca0f51bf1d3b0e33eaeadf315a1362f68bf9c51dc9db1be002bfc` | `5b73a5eeaf7134b2afb8d5f1b9db7baf5c63e99c3b966c7f42d04d1555656d00` | 27 | path and anonymisation changes only |
| `experiments/transfer/task_x.py` | `task_x.py` | `312d3e6c61e277ef258dc89713862a6c09f1c5c35fe05bc5fd775e681d638c9a` | `312d3e6c61e277ef258dc89713862a6c09f1c5c35fe05bc5fd775e681d638c9a` | 0 | byte identical |
| `experiments/relative_dose/r_voc_analyze.py` | `r_voc_analyze.py` | `37952f01c2343cb964613bb448065763e9c0dbf4a527b5a51088926797cd5eea` | `d353d939a1c99dc44251b0b84e5b599462ec2864508d3920385d44f1343db304` | 20 | path and anonymisation changes only |
| `experiments/relative_dose/task_r.py` | `task_r.py` | `2f6f986e4b0a2659cead9871f170b8fee7c1fa2dd09af7b8cb22694552a59859` | `fb59f644ab448775855d36167b2f61b33609e25f3f01faa3033019084b0e3a5a` | 23 | path and anonymisation changes only |
| `experiments/fixed_area/n_voc_analyze.py` | `n_voc_analyze.py` | `a9e1e6ba2836205a24ad1738e836b6cb43e793ed8e4375fbb36e856798cd44cd` | `e7b5be2928886c4df8bab6ec63bed2de6d2017438c21190989ae2a8ea81ba7e5` | 26 | path and anonymisation changes only |
| `experiments/fixed_area/task_n2_deim_official.py` | `task_n2_deim_official.py` | `1e5929770c875d9bc40c95ed3ee8accbac23045627f71032f527d6f3557a9d95` | `b23aabf4fab99d5bcffbc3ba72a219b4019aa9ca9a142d239053b76ecbf96d55` | 35 | path and anonymisation changes only |
| `experiments/deformable/task_d3.py` | `task_d3.py` | `fd11ec4943c83a1fe34069388dff46df5514fdfc249227b0313e36d31de64776` | `ef2cc7934405cc61f630cbf1e13b73f02fe2e84c972cac3f5f27e03a237f230f` | 23 | path and anonymisation changes only |
| `experiments/model_comparison/recompute_y.py` | `recompute_y.py` | `617cc11c2df734a58691459bd89a7238756d38e01153f317a559dce6ae9b6e75` | `617cc11c2df734a58691459bd89a7238756d38e01153f317a559dce6ae9b6e75` | 0 | byte identical |
| `experiments/model_comparison/task_y.py` | `task_y.py` | `87970c6cec21486a74d8e1daba0bd1e37e7d8d2d5e91d654c91986741adf5c17` | `092c43fe457929b2e391d58e5bad89ea4da082c11303d29760f9ca41cd933276` | 19 | path and anonymisation changes only |
| `experiments/model_comparison/task_y_geo.py` | `task_y_geo.py` | `c01c24779dc9b8821e07586d6a4a8dc42f33fc6e4d498fe5d477e2dd2b3303f5` | `0ebd3716cfe5d026aff0a279530eed26dd890d5cb2030518c6008e7413a55a17` | 19 | path and anonymisation changes only |
| `experiments/model_comparison/task_y_paired.py` | `task_y_paired.py` | `4ed348ebeb9aa7e942a9ae1b4766fe34cb5b9421c0a01655b44777adb131feb5` | `d47e2cefa42f56e3c3a13108890dda26a48ad93dc522cf76d000a000862b4064` | 23 | path and anonymisation changes only |
| `experiments/model_comparison/task_y_stats.py` | `task_y_stats.py` | `6d9f15a144ac6f31acd1cf1dd242597dfcc88f2887e887248c1bca327a477745` | `5bc9bc03881eebb602c6a2704459c52dd6e98a7c3b35f888e89504b743c149b4` | 25 | path and anonymisation changes only |
| `experiments/detector_ap/task_c_v2.py` | `task_c_v2.py` | `9d9acd465065bd236022abf55dbfc027649d313fa14b4c0965f6b16a507b3104` | `870f6bd064dcc90090df2e2cb9b94e695809b91b8ddccbb1da1f8afb481cf984` | 19 | path and anonymisation changes only |
| `experiments/local_quality/recompute_tables.py` | `recompute_tables.py` | `32440a0a1ad1f747966fc334aeecd9c94e522b0698c04f37528f9bff01ab93ca` | `90556ce6b13f1df691bf7c97a11744a169c2b1e7beaf2043a1c7afe5ae596562` | 19 | path and anonymisation changes only |

## Registration texts (`registrations/`)

| released file | original file | original SHA256 | released SHA256 | changed lines | note |
|---|---|---|---|---|---|
| `registrations/preregistration_coco.md` | - | - | `6ef4f654fb028f7658eaeadf00ded7a64b7c8dcbd02f4bb9b7d31e8ca6317232` | - | frozen text, byte identical to the copy kept with the submission; the original file is outside this release |
| `registrations/registration_voc2007.md` | - | - | `040a5053ae98091ed516b2231ab2939ec4e2726f080b2e06e893ee9ba34aae5a` | - | new file written for the release |
| `registrations/addendum_G.md` | - | - | `5d0d995ad379fad37ec555ea77cf50e9f225f3f0095589e29f00e9803fb68797` | - | frozen text, byte identical to the copy kept with the submission; the original file is outside this release |
| `registrations/addendum_N.md` | - | - | `f3d31ce7dc7ce8f2e92908b7a6359c458cdd478b6d5d91f905f8f886d4b5ad9d` | - | frozen text, byte identical to the copy kept with the submission; the original file is outside this release |
| `registrations/addendum_D3.md` | - | - | `6952c33371ad7705cc5a026308e4173f67ae7233dba5ca4872f8c2c344b9b791` | - | frozen text, byte identical to the copy kept with the submission; the original file is outside this release |
| `registrations/addendum_G5.md` | `CONFIRM_REG_v1_addendum_G5.md` | `a346a5d10d74196ed3f55f4dc34348abad4787a2164703a2fc51561376c25a50` | `a346a5d10d74196ed3f55f4dc34348abad4787a2164703a2fc51561376c25a50` | 0 | byte identical |
| `registrations/addendum_R.md` | `CONFIRM_REG_v1_addendum_R.md` | `73b81f851fcab458db8c781b1320ac6fcb72a6948329d74bc6dfc6ed616ebfbc` | `73b81f851fcab458db8c781b1320ac6fcb72a6948329d74bc6dfc6ed616ebfbc` | 0 | byte identical |
| `registrations/addendum_Y.md` | `CONFIRM_REG_v1_addendum_Y.md` | `925b36bceb9e6ed25ce85ac6063e658cb176cf7fa48807c55bf3d4fe2efca441` | `925b36bceb9e6ed25ce85ac6063e658cb176cf7fa48807c55bf3d4fe2efca441` | 0 | byte identical |
| `registrations/addendum_Y2.md` | `CONFIRM_REG_v1_addendum_Y2.md` | `206b947672838c3b00e885ed8906aa0241c8858ac41b3e4e9bda6c88df2544ea` | `206b947672838c3b00e885ed8906aa0241c8858ac41b3e4e9bda6c88df2544ea` | 0 | byte identical |

## Analysis entry points written for the release

`scripts/reproduce_tables.py` and `scripts/verify_release.py` are new: they call
the frozen scripts above in a fixed order and compare the result with the
published numbers. They do not implement any statistic themselves.

## Configuration files

* `configs/models.yaml` `5698faf37b35d0020e5b2b22cc2ef051a9542851a1dc201a4f06a1dc5831f0a7`
* `requirements.txt` `172646cf383c9866ceb82b10799cbbb42dbaab8214d37c9f58e0bb5e44c6b0f6`
* `requirements-cpu.txt` `ca223e20db7aad5a87341d575162b64827d6fae7ece192364fa96b61d26e40b9`
* `environment.yml` `262ddd3cbc8c715b3e63bca90347ef938b7ad71fb8c4fd0a22853a347a7c751e`

