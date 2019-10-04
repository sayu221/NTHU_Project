  class ResnetkBasicBlock(nn.module):
        # Still need to be cheked 
        def __init__(self, in_c, out_c, kernel_size=3, scale=1.0, activation_fn=tf.nn.relu):
            super(DilatedBasicBlock, self).__init__()
            #num = output_channel
            #num_out = num
            #num2 = (num_out >> 1)
            self.conv1 = slimconv2d(in_c, out_c/2, kernel_size=1, stride=1)     # tower0 = slim.conv2d(net, num2, 1, stride=1)
            self.conv2 = slim.conv2d(in_c, out_c/2, kernel_size=1, stride=1)   # tower1 = self.conv2d(net, num2, 1, stride=1)
            self.conv3 = slim.conv2d(out_c/2, out_c/2, kernel_size= [1, kernel_size], stride=1) # tower1 = self.conv2d(tower1, num2, [1, kernel_size], stride=1)
            self.conv4 = slim.conv2d(out_c/2, out_c/2, kernel_size= [kernel_size, 1], stride=1) # tower1 = self.conv2d(tower1, num2, [kernel_size, 1], stride=1)
            self.conv5 = conv2d(out_c/2, in_c, kernel_size=1, stride=1)  # mixup = self.conv2d( mixed, num_out, 1, stride=1, normalizer_fn=None, activation_fn=None)
            self.scale = scale
            self.relu = nn.ReLU(inplace=True)
   
        def forward(self, x):
    
            tower0 = self.conv1(x)
            tower1 = self.conv2(x)
            tower1 = self.conv3(tower1)
            tower1 = self.conv4(tower1)
            
            mixed = torch.cat((self.conv1, self.conv4), 1)  # mixed = self.concat(axis=-1, values=[tower0, tower1])
            mixup = self.conv2d(mixed)

            out += mixup * self.scale
            out = self.relu(out)
            
        return out


