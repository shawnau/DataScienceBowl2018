# ------------------------------------------------------
# def instance_to_ellipse(instance):
#  contour = thresh & thresh_to_contour(thresh, radius=1)
#     y,x = np.where(contour)
#     points =  np.vstack((x, y)).T.reshape(-1,1,2)
#
#     #<todo> : see if want to include object that is less than 5 pixel area? near the image boundary?
#     if len(points) >=5:
#         (cx, cy), (minor_d, major_d), angle = cv2.fitEllipse(points)
#         minor_r = minor_d/2
#         major_r = major_d/2
#
#         minor_r = max(1,minor_r)
#         major_r = max(1,major_r)
#
#         #check if valid
#         tx, ty = get_center(thresh)
#         th, tw = thresh.shape[:2]
#         distance = ((cx-tx)*(cx-tx) + (cy-ty)*(cy-ty))**0.5
#         if distance < (major_r+ minor_r)*0.5 and cx>1 and cx<tw-1 and cy>1 and cy <th-1:
#             return cx, cy, minor_r,major_r,  angle
# # r=[8,16,32]
# def multi_mask_to_center(multi_mask, scales=[2,4,8,16]):
#
#     # 3 = math.log2( 8)
#     # 5 = math.log2(32)
#     limits = [  math.log2(s) for s in scales ]
#     limits = [ [c-0.7,c+0.7] for c in limits ]
#     limits[-1][0] = 0
#     limits[ 1][1] = math.inf
#     num = len(limits)
#
#
#     H,W = multi_mask
#     xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
#
#     center = np.zeros((num, H,W), np.bool)
#     delta  = np.zeros((5,H,W), np.float32)  #cx, cy, major_r, minor_r, angle
#
#
#     num_masks = multi_mask.max()
#     for i in range(num_masks):
#         instance = num_masks==(i+1)
#
#         ret = thresh_to_ellipse(thresh)
#         if ret is None: continue
#
#         cx, cy, minor_r, major_r,  angle = ret
#         r = (minor_r + major_r)/2
#         t = math.log2(r)
#
#         cr = max(1,int(0.5*r))
#         c = np.zeros((H,W), np.uint8)
#         cv2.circle(c, (int(cx),int(cy)), int(cr), 255, -1) # cv2.LINE_AA)
#         c = c>128
#
#
#         for j,limit in enumerate(limits):
#             if  t >limit[0] and t<limit[1]:
#                 center[j] = center[j] | c
#
#
#         #delta
#         index = np.where(c)
#         delta[0][index] = xx[index]-cx
#         delta[1][index] = yy[index]-cy
#         delta[2][index] = minor_r
#         delta[3][index] = major_r
#         delta[4][index] = angle
#
#
#     return center, delta