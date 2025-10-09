## SAQE: Complex Logical Query Answering via Semantic-Aware Representation Learning




 Obtain the pre-trained backbone models

```bash
$ mkdir SAQE-M/models/
$ for i in "fb15k" "fb15k-237" "nell"; do for j in "betae" "q2b"; do wget -c http://data.neuralnoise.com/kgreasoning-cqd/$i-$j.tar.gz; done; done
$ for i in *.tar.gz; do tar xvfz $i; done
```

To train the SAQE model, run the following commands under the `SAQE/` folder.

```bash
$ cd kbc/src
$ bash ../scripts/preprocess.sh
$ bash ../scripts/train_[dataset].sh 
$ bash ../scripts/train_fb15k237.sh

$ cd ../..
$ bash scripts/build_neural_adj_15k.sh
$ bash scripts/build_neural_adj_237.sh
$ bash scripts/build_neural_adj_nell.sh
```

```bash
$ cd ..
$ bash scripts/valid_memb_func.sh SAQE FB15k-237 0 
$ bash scripts/valid_memb_func.sh SAQE FB15k 0 
$ bash scripts/valid_memb_func.sh SAQE NELL 0 

$ bash scripts/grid_valid.sh SAQE FB15k-237 0 
$ bash scripts/grid_valid.sh SAQE FB15k 1 
$ bash scripts/grid_valid.sh SAQE NELL 1 

$ bash scripts/eval_fs_fb15k-SAQE.sh > log/fb15k-SAQE.log
$ bash scripts/eval_fs_fb15k-237-SAQE.sh > log/fb15k-237-SAQE.log
$ bash scripts/eval_fs_nell-SAQE.sh > log/nell-SAQE.log
```


If you have any question, please contact to caozongsheng@iie.ac.cn

