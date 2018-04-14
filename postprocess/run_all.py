from predict_boxes import run_predict
from ensemble_boxes import run_ensemble_box
from predict_masks import run_predict_mask_only
from ensemble_masks import ensemble_masks

if __name__ == '__main__':
    run_predict()
    run_ensemble_box()
    run_predict_mask_only()
    ensemble_masks()