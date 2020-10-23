import cv2 as cv
import argparse
import numpy as np





xstart = 0
xend = 0
ystart = 0
yend = 0

   
def getMemoryShapes(object,inputs):
    inputShape, targetShape = inputs[0], inputs[1]
    batchSize, numChannels = inputShape[0], inputShape[1]
    height, width = targetShape[2], targetShape[3]

    ystart = (inputShape[2] - targetShape[2]) // 2
    xstart = (inputShape[3] - targetShape[3]) // 2
    yend = ystart + height
    xend = xstart + width

    return [[batchSize, numChannels, height, width]]

def forward(object,inputs):
    getMemoryshapes(object,inputs)
    return [inputs[0][:,:,ystart:yend,xstart:xend]]

cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv.dnn.readNet(args.prototxt, args.caffemodel)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'
cv.namedWindow(kWinName, cv.WINDOW_AUTOSIZE)

cap = cv.VideoCapture(args.input if args.input else 0)

if args.write_video:
    # Define the codec and create VideoWriter object
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(w,h)
    # w, h = args.width,args.height
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    writer = cv.VideoWriter(args.savefile, fourcc, 25, (w, h))
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)
    con=np.concatenate((frame,out),axis=1)
    if args.write_video:
        writer.write(np.uint8(con))
    cv.imshow(kWinName,con)