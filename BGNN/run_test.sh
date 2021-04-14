#!/usr/bin/env bash

# BGNN (Adv)
#nohup sh run_bgnn_adv.sh tencent > log_run_abcgraph_adv_tencent.txt 2>&1 &
nohup sh run_bgnn_adv.sh cora > log_run_abcgraph_adv_cora.txt 2>&1 &
nohup sh run_bgnn_adv.sh citeseer > log_run_abcgraph_adv_citeseer.txt 2>&1 &
nohup sh run_bgnn_adv.sh pubmed > log_run_abcgraph_adv_pubmed.txt 2>&1 &