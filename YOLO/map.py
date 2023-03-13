import numpy as np

def calculate_iou(box1, box2):
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  area1 = w1 * h1
  area2 = w2 * h2
  
  x_intersect = max(0, min(x1+w1, x2+w2) - max(x1, x2))
  y_intersect = max(0, min(y1+h1, y2+h2) - max(y1, y2))
  area_intersect = x_intersect * y_intersect
  
  iou = area_intersect / (area1 + area2 - area_intersect)
  return iou

def calculate_ap(recall, precision):
  ap = 0
  for i in range(len(recall)-1):
    ap += (recall[i+1] - recall[i]) * ((precision[i+1] + precision[i]) / 2)
  return ap

def calculate_map(y_true, y_pred, iou_threshold = np.linspace(0.5, 0.95, 1)):
  n_classes = y_true.shape[1] - 4
  ap_list = np.zeros(n_classes)
  
  for c in range(n_classes):
    y_pred_c = y_pred[y_true[:, c+4] == 1]
    y_true_c = y_true[y_true[:, c+4] == 1]
    
    n_grond_truths = len(y_true_c)
    n_predictions = len(y_pred_c)
    
    if n_grond_truths == 0:
      continue
    
    if n_predictions == 0:
      ap_list[c] = 0
      continue
    
    y_pred_c = y_pred_c[np.argsort(y_pred_c[:, 0])[::-1]]
    
    tp = fp = np.zeros(n_predictions)
    
    for i in range(n_predictions):
      box_pred = y_pred_c[i, 1:5]
      iou_max = -1
      ground_truth_match = -1
      
      for j in range(n_grond_truths):
        if y_true_c[j, 0] != c:
          continue
        
        box_true = y_true_c[j, 1:5]
        iou = calculate_iou(box_pred, box_true)
        
        if iou > iou_max:
          iou_max = iou
          ground_truth_match = j
          
      if iou_max > iou_threshold:
        if y_true_c[ground_truth_match, 2] == 0:
          tp[i] = 1
          y_true_c[ground_truth_match, 2] = i
        else:
          fp[i] = 1
      else:
        fp[i] = 1
        
      tp_cumsum = np.cumsum(tp)
      fp_cumsum = np.cumsum(fp)
      
      recall = tp_cumsum / n_grond_truths
      precision = tp_cumsum / (tp_cumsum + fp_cumsum)
      
      ap_list[c] = calculate_ap(recall, precision)
    
    map = np.mean(ap_list)
    return map
    
def test():
  y_true = np.array([
    [0, 10, 10, 20, 20, 1, 0],
    [0, 30, 30, 40, 40, 1, 0],
  ])  
  y_pred = np.array([
    [0.7, 10, 10, 20, 20],
    [0.8, 30, 30, 40, 40],
  ])
  
  map_list = calculate_map(y_true, y_pred)
  print(map_list)
  
if __name__ == '__main__':
  test()