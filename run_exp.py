# import pdb; pdb.set_trace()
import argparse
import torch
import os
import csv
import json
import numpy as np

from pulseppg.utils.utils import printlog,  init_dl_program, count_parameters
from pulseppg.utils.imports import import_model
from pulseppg.utils.datasets import load_data

from pulseppg.experiments.configs.MotifDist_expconfigs import allmotifdist_expconfigs
from pulseppg.experiments.configs.PulsePPG_expconfigs import allpulseppg_expconfigs

all_expconfigs = {**allmotifdist_expconfigs, **allpulseppg_expconfigs}

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--retrain_eval", help="WARNING: Retrain eval model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--test_idx", help="Choose specific index to output predicted value from",
                        default=None)
    parser.add_argument("--dontprint", help="Printing verbose output",
                        action='store_true', default=False)
    parser.add_argument("--resume_on", help="resume unfinished model training",
                        action='store_true', default=False)
    parser.add_argument("--pre_pilot_eval", help="Preprocess pre-pilot data and export PulsePPG embeddings without training",
                        action='store_true', default=False)
    parser.add_argument("--pre_pilot_cluster", help="Cluster pre-pilot embeddings and create timeline visualizations",
                        action='store_true', default=False)
    parser.add_argument("--pre_pilot_bpm", help="Apply a trained BPM/HR regression probe to pre-pilot embeddings",
                        action='store_true', default=False)
    parser.add_argument("--pre_pilot_overwrite", help="Overwrite existing pre-pilot preprocessing output before exporting embeddings",
                        action='store_true', default=False)
    parser.add_argument("--pre_pilot_checkpoint", help="Checkpoint name to use for pre-pilot embedding export",
                        default="best")
    parser.add_argument("--pre_pilot_output_dir", help="Directory for pre-pilot embedding outputs",
                        default="pulseppg/data/datasets/pre_pilot_outputs")
    parser.add_argument("--pre_pilot_max_chunks", help="Optional cap for pre-pilot embedding export smoke tests",
                        type=int, default=None)
    parser.add_argument("--pre_pilot_n_clusters", help="Number of clusters for pre-pilot embedding clustering",
                        type=int, default=6)
    parser.add_argument("--bpm_probe_dir", help="Directory containing checkpoint_scaler_best.joblib and checkpoint_cv_best.joblib",
                        default="pulseppg/experiments/out/pulseppg/PPG-DaLiA | HR | Linear Probe/linear_probe")
    parser.add_argument("--device", help="Device for pre-pilot embedding export, e.g. cuda or cpu",
                        default=None)
    args = parser.parse_args()


    # selecting config according to arg
    # CONFIGFILE = "24_1_4_ppgdist_stride2_5maskperc50"
    CONFIGFILE = args.config
    config = all_expconfigs[CONFIGFILE]
    config.set_rundir(CONFIGFILE)

    if args.pre_pilot_eval or args.pre_pilot_cluster or args.pre_pilot_bpm:
        from pathlib import Path
        from pulseppg.data.process.PRE_PILOT import (
            cluster_pre_pilot_embeddings,
            export_pre_pilot_embeddings,
            predict_pre_pilot_bpm,
        )

        summary = {}
        if args.pre_pilot_eval:
            summary["embedding_export"] = export_pre_pilot_embeddings(
                model_config_key=CONFIGFILE,
                checkpoint=args.pre_pilot_checkpoint,
                output_dir=args.pre_pilot_output_dir,
                device=args.device,
                preprocess_overwrite=args.pre_pilot_overwrite,
                max_chunks=args.pre_pilot_max_chunks,
            )
        if args.pre_pilot_cluster:
            output_dir = Path(args.pre_pilot_output_dir)
            summary["clustering"] = cluster_pre_pilot_embeddings(
                embeddings_path=output_dir / f"{CONFIGFILE}_{args.pre_pilot_checkpoint}_embeddings.npz",
                index_path=output_dir / f"{CONFIGFILE}_{args.pre_pilot_checkpoint}_embedding_index.csv",
                output_dir=output_dir / "clusters",
                n_clusters=args.pre_pilot_n_clusters,
            )
        if args.pre_pilot_bpm:
            output_dir = Path(args.pre_pilot_output_dir)
            summary["bpm_prediction"] = predict_pre_pilot_bpm(
                embeddings_path=output_dir / f"{CONFIGFILE}_{args.pre_pilot_checkpoint}_embeddings.npz",
                index_path=output_dir / f"{CONFIGFILE}_{args.pre_pilot_checkpoint}_embedding_index.csv",
                probe_dir=args.bpm_probe_dir,
                output_dir=output_dir / "bpm",
            )
        print(json.dumps(summary, indent=2))
        raise SystemExit(0)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    train_data, train_labels, val_data, val_labels, test_data, test_labels  = \
        load_data(data_config = config.data_config)

    model = import_model(config, 
                        train_data=train_data, train_labels=train_labels, 
                        val_data=val_data, val_labels=val_labels, 
                        test_data=test_data, test_labels=test_labels, 
                        resume_on = args.resume_on)
    
    table, total_params = count_parameters(model.net)
    print(f"Total Trainable Params: {total_params:,}")

    try:
        logpath = os.path.join("pulseppg/experiments/out", config.run_dir)
        printlog(f"----------------------------------------------------------------------------------- Config: {CONFIGFILE} -----------------------------------------------------------------------------------", logpath)

        if (args.retrain == True) or (not os.path.exists(os.path.join("pulseppg/experiments/out/", 
                                                                config.run_dir, 
                                                                "checkpoint_best.pkl"))):
            model.fit()

        all_eval_results_title = ["name", "notes"]
        all_eval_results = [CONFIGFILE, f"{total_params:,}"]
        for eval_config in config.eval_configs:
            printlog(f"Starting {eval_config.name} evaluation", logpath)

            out_test_all = []

            train_data, train_labels, val_data, val_labels, test_data, test_labels = \
                load_data(data_config = eval_config.data_config)
            
            eval_config.set_rundir(os.path.join(CONFIGFILE, eval_config.name, eval_config.model_file))
            # loading eval model
            evalmodel = import_model(eval_config, 
                                    train_data=train_data, train_labels=train_labels, 
                                    val_data=val_data, val_labels=val_labels, 
                                    test_data=test_data, test_labels=test_labels,
                                    # reload checkpoint is off bc we are loading just the eval model
                                    reload_ckpt = False, 
                                    evalmodel=True)
            # loading pre-trained model
            model = import_model(config, reload_ckpt=eval_config.pretrain_epoch)
            # adds pre-trained model to eval model
            evalmodel.setup_eval(trained_net=model.net)

            if (args.retrain_eval == True) or (not os.path.exists(os.path.join(evalmodel.run_dir, "checkpoint_best.pkl"))):
                evalmodel.fit()

            if args.test_idx is not None:
                args.test_idx=int(args.test_idx)
            out_test = evalmodel.test(test_idx=args.test_idx, dontprint=args.dontprint) # automatically loads
            printlog(eval_config.name + " " + eval_config.model_file +" ++++++++++++++++++++++++++++++++++++++++", logpath)

            all_eval_results_title.extend(list(out_test.keys()))
            all_eval_results.extend(list(out_test.values()))

        # create csv file that is easy to paste into spreadsheet
        csv_file = os.path.join(logpath, f"{CONFIGFILE}_easy_paste.csv")
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(all_eval_results_title)
            writer.writerow(all_eval_results)
            
    except Exception as e:
        raise  
    finally:
        printlog(f"Config: {CONFIGFILE}", logpath)
