# def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
#     # image * α + mask * β + λ
#
#     if threshold is None:
#         mask = mask/255
#     else:
#         mask = clean_mask(mask,threshold,1)
#
#     mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
#     image[...] = cv2.addWeighted(image, α, mask, β, λ)
#


# def draw_contour(image, mask, color=(0,255,0), thickness=1, threshold=127):
#     ret, thresh = cv2.threshold(mask,threshold,255,0)
#     ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     hierarchy = ret[0]
#     contours  = ret[1]
#     #image[...]=image
#     cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)
#     ## drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None): # real signature unknown; restored from __doc__
#
#

# def draw_rcnn_pre_nms(image, probs, deltas, proposals, cfg, colors, names, threshold=-1, is_before=1, is_after=1):
#
#     height,width = image.shape[0:2]
#     num_classes  = cfg.num_classes
#
#     probs  = probs.cpu().data.numpy()
#     deltas = deltas.cpu().data.numpy()
#     proposals    = proposals.data.cpu().numpy()
#     num_proposals = len(proposals)
#
#     labels = np.argmax(probs,axis=1)
#     probs  = probs[range(0,num_proposals),labels]
#     idx    = np.argsort(probs)
#     for j in range(num_proposals):
#         i = idx[j]
#
#         s = probs[i]
#         l = labels[i]
#         if s<threshold or l==0:
#             continue
#
#         a = proposals[i, 0:4]
#         t = deltas[i,l*4:(l+1)*4]
#         b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
#         b = clip_boxes(b, width, height)  ## clip here if you have drawing error
#         b = b.reshape(-1)
#
#         #a   = a.astype(np.int32)
#         color = (s*np.array(colors[l])).astype(np.uint8)
#         color = (int(color[0]),int(color[1]),int(color[2]))
#         if is_before==1:
#             draw_dotted_rect(image,(a[0], a[1]), (a[2], a[3]), color, 1)
#             #cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (int(color[0]),int(color[1]),int(color[2])), 1)
#
#         if is_after==1:
#             cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color, 1)
#
#         draw_shadow_text(image , '%f'%s,(b[0], b[1]), 0.5, (255,255,255), 1, cv2.LINE_AA)
#