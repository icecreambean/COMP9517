


border = int(max(g.shape)/4)
g_new = cv2.copyMakeBorder(g, border, border, border, border, cv2.BORDER_CONSTANT)

res = cv2.matchTemplate(r,g_new,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
r_top_left = max_loc
r_bottom_right = (r_top_left[0] + r.shape[1], r_top_left[1] + r.shape[0])

res = cv2.matchTemplate(b,g_new,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
b_top_left = max_loc
b_bottom_right = (b_top_left[0] + b.shape[1], b_top_left[1] + b.shape[0])

r_new = np.zeros(g_new.shape)
print(r_new.shape, r_top_left, r.shape)

h,w = r.shape
rstart, cstart = r_top_left
rstop = rstart + w
cstop = cstart + h
print(r_new[rstart:rstop,cstart:cstop].shape)
print(rstart, cstart, rstop, cstop)

r_new[cstart:cstop,rstart:rstop] = r

cv2.rectangle(r_new,r_top_left, r_bottom_right, 255, 2)

#bottom_right = (top_left[0] + w, top_left[1] + h)
#cv2.rectangle(b,top_left, bottom_right, 255, 2)

#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

#color_img = cv2.merge((b,g,r))
#plt.imshow(color_img)

plt.subplot(1,2,1)
plt.imshow(r_new)
plt.subplot(1,2,2)
plt.imshow(g_new)

r_top_left, b_top_left






bn,gn,rn = alignComponents(b,g,r)
plt.subplot(3,1,1)
plt.imshow(bn)
plt.subplot(3,1,2)
plt.imshow(gn)
plt.subplot(3,1,3)
plt.imshow(rn)
bn.shape,gn.shape,rn.shape



plt.subplot(3,1,1)
plt.imshow(b)
plt.subplot(3,1,2)
plt.imshow(g)
plt.subplot(3,1,3)
plt.imshow(r)




ptype = type(g[0,0])
img_plane = np.array([np.iinfo(ptype).max] * template.size)
img_plane = np.reshape(img_plane, template.shape)


# comparison logic
#def _matchTemplate(mt_img, mt_template, cv_method):
#    mt_result = cv2.matchTemplate(mt_img, mt_template, cv_method)
#    _, _, mt_locmin, mt_locmax = cv2.minMaxLoc(mt_result)
#    if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#        mt_start = mt_locmin
#    else:
#        mt_start = mt_locmax
#    mt_stop  = (mt_start[0]+width(mt_img), mt_start[1]+height(mt_img))
#    return mt_start, mt_stop # (top left, bot right) each in (r,c) format

# perform image alignment
def alignComponents(b,g,r):
    # assume g == template (most accurately sliced)
    border = int(max(g.shape)/4)
    g_new = cv2.copyMakeBorder(g, border, border, border, border, cv2.BORDER_CONSTANT)
    pix_type = type(g[0,0])
    
    # compare: r,b <--> g (template)
    r_start, _ = _matchTemplate(r, g_new)
    b_start, _ = _matchTemplate(b, g_new)
    
    # apply translation (to larger img plane)
    r_new = np.zeros(g_new.shape, dtype=pix_type)
    b_new = np.zeros(g_new.shape, dtype=pix_type)
    
    print(r_start, b_start)
    
    r_new[ r_start[1]:r_start[1]+height(r), r_start[1]:r_start[1]+width(r) ] = r
    b_new[ b_start[0]:b_start[0]+height(b), b_start[1]:b_start[1]+width(b) ] = b
    
    img_out = cv2.merge((b_new,g_new,r_new))
    return img_out




mt: off(54,54): imgsize=(106, 167) (maxsize=(160, 221)); res=1290.5042508521697