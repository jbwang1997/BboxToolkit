import numpy as np


def cat(bbox_list, axis=0, score_full=1.):
    if not bbox_list:
        raise ValueError('bbox_list is empty!')

    new_bboxes = []
    new_scores = []
    with_scores = sum([B.with_scores for B in bbox_list])
    for B in bbox_list:
        bboxes_np = B.bboxes
        scores_np = B.scores
        new_bboxes.append(bboxes_np)
        if with_scores:
            if scores_np is None:
                scores_np = np.zeros(bboxes_np.shape[:-1],
                                     dtype=np.float32) + score_full
            new_scores.append(scores_np)

    new_bboxes = np.concatenate(new_bboxes, axis=axis)
    new_scores = np.concatenate(new_scores, axis=axis) \
            if with_scores else None

    return type(bbox_list[0])(new_bboxes, new_scores)
