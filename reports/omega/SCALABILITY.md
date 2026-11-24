# Single-process reconstruction costs

| event | load s | graph s | seed assembly s | alternative pool s | candidate choice s | EKF extension s | metric evaluation s | peak MiB |
|---|---|---|---|---|---|---|---|---|
| event000001027 | 0.262055 | 0.171231 | 0.784426 | 0.229877 | 0.083922 | 0.223348 | 0.392763 | 369.468750 |
| event000001185 | 0.368419 | 0.383179 | 0.962507 | 0.158059 | 0.161122 | 1.027817 | 1.626726 | 369.468750 |
| event000001343 | 0.300380 | 0.246697 | 0.859605 | 0.144005 | 0.110646 | 0.267415 | 0.355899 | 369.468750 |
| event000001502 | 0.248384 | 0.160477 | 0.976919 | 0.284966 | 0.149377 | 0.327108 | 0.457651 | 369.468750 |
| event000001660 | 0.361523 | 0.270948 | 1.255944 | 0.351686 | 0.084320 | 0.192776 | 0.344482 | 369.468750 |
| event000001818 | 0.241834 | 0.154441 | 0.737733 | 0.272289 | 0.088346 | 0.234451 | 0.502080 | 369.468750 |
| event000001977 | 0.322230 | 0.205630 | 0.785565 | 0.132711 | 0.069506 | 0.175101 | 0.447268 | 369.468750 |
| event000002135 | 0.328698 | 0.203484 | 0.870191 | 0.194821 | 0.099072 | 0.221870 | 0.408928 | 369.468750 |
| event000002293 | 0.305603 | 0.189657 | 0.910427 | 0.183244 | 0.119444 | 0.343085 | 0.494122 | 369.468750 |
| event000002502 | 0.410734 | 0.202882 | 1.036638 | 0.169682 | 0.117560 | 0.240741 | 0.597658 | 369.468750 |
| event000002660 | 0.218788 | 0.182832 | 0.825555 | 0.228256 | 0.081536 | 0.193153 | 0.326488 | 369.468750 |
| event000002819 | 0.267495 | 0.145604 | 0.574006 | 0.122640 | 0.065295 | 0.165732 | 0.302624 | 369.468750 |

Runtime in comparisons includes common two-pass seeding, geometry indexing, method-specific selection and extension. Validation, reference scoring and truth diagnostics run outside reconstruction. Evaluation timing combines matching and scoring. RSS is the process-lifetime high-water mark across both arms.

| rank microbenchmark | unique selected | scalar median s | vector median s |
|---|---|---|---|
| event000001003 | 4324 | 0.297033 | 0.022831 |
| event000001004 | 5828 | 0.273076 | 0.016057 |
| event000001005 | 4504 | 0.306727 | 0.031084 |

The batched same-length residual kernel exactly reproduced the ordered seed list in four alternating timed repetitions per event. Only the resolver was timed in that comparison. The confirmation used the frozen scalar baseline. The distinct end-to-end experiment below times geometry indexing, both sector passes, replacement and EKF extension. It omits an unused intermediate resolver pass and batches the final ranking.

| development event | frozen median wall s | batched median wall s | ordered paths |
|---|---|---|---|
| event000001003 | 1.856437 | 1.587415 | 4324 |
| event000001004 | 2.860616 | 2.275097 | 5828 |
| event000001005 | 1.607887 | 1.323349 | 4504 |

Three alternating runs per implementation per event yielded exactly the same raw seeds, selected seeds, final hit sequences and official score. This is a historical development timing study. The twelve-event primary comparison retains its originally measured costs.
