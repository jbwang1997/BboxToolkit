import cv2


def imshow(img, win_name='', wait_time=0, max_size=1000):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        max_size (int): Max size of window
    """
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        win_height = max_size if height > width else int(max_size*height/width)
        win_width = max_size if width >= height else int(max_size*width/height)
    else:
        win_height, win_width = height, width
    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, win_width, win_height)
    cv2.imshow(win_name, img)
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  class_names=None,
                  score_thr=0,
                  draw_ppls=None,
                  colors='green',
                  bbox_tk=1,
                  text_tk=5,
                  show=True,
                  win_name='',
                  wait_time=0,
                  max_size=1000,
                  out_file=None):



    if show:
        imshow(img, win_name, wait_time, max_size)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img
