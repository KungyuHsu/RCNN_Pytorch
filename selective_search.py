import cv2
class SelectiveSearch:
    def __init__(self,mode='f'):
        self.gs=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.mode=mode
    def __call__(self, img):
        #配置参数
        self.gs.setBaseImage(img)
        if self.mode=='s':
            self.gs.switchToSingleStrategy()
        elif self.mode=='f':
            self.gs.switchToSelectiveSearchFast()
        elif self.mode=='q':
            self.gs.switchToSelectiveSearchQuality()
        #运行算法
        rects = self.gs.process()
        rects[:, 2] += rects[:, 0]
        rects[:, 3] += rects[:, 1]
        return rects

if __name__ == '__main__':
    img = cv2.imread("./data/test.jpg",cv2.IMREAD_COLOR)
    ss=SelectiveSearch()
    rects=ss(img)
    print(rects)
    print("tlt len:",len(rects))
    for i,rect in enumerate(rects):
        if i>20:
            break
        else:
            x1,y1,x2,y2=rect
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("testimg",img)
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyWindow("testimg")
