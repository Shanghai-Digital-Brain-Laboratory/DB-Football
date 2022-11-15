# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

class NamedIndex:
    def __init__(self):
        # feat_name: (s_idx,coder)
        self.name_idx_mapping={}
        self.tags={}
        self._len=0

    def __len__(self):
        return self._len

    def feat_len(self,name):
        s_idx,e_idx,_=self.name_idx_mapping[name]
        return e_idx-s_idx

    def tag(self,new_name,names):
        list_index=self.get_list_index(names)
        self.tags[new_name]=list_index

    def register(self,name,coder_or_len):
        assert name not in self.name_idx_mapping
        s_idx=self._len
        if type(coder_or_len)==int:
            e_idx=self._len+coder_or_len
            coder=None
        else:
            coder=coder_or_len
            e_idx=self._len+coder._len
        self.name_idx_mapping[name]=(s_idx,e_idx,coder)
        self._len=e_idx
        self.name_idx_mapping["__all__"]=(0,self._len,None)

    def get_index(self,name):
        s_idx,e_idx,_=self.name_idx_mapping[name]
        return s_idx,e_idx

    def get_slice_index(self,name):
        s_idx,e_idx=self.get_index(name)
        return slice(s_idx,e_idx)

    def get_list_index(self,names):
        indices=[]
        for name in names:
            s_idx,e_idx,_=self.name_idx_mapping[name]
            indices+=list(np.arange(s_idx,e_idx))
        return indices

    def get_list_index_by_tag(self,name):
        return self.tags[name]

    def zeros(self):
        return np.zeros(len(self),dtype=float)

    def write(self,feat,name,feat_):
        assert feat[...,self.get_slice_index(name)].size==feat_.size,"{} vs {}".format(feat[...,self.get_slice_index(name)].size,feat_.size)
        feat[...,self.get_slice_index(name)]=feat_.reshape(feat[...,self.get_slice_index(name)].shape)

    def onehot(self,idx,n):
        feat=np.zeros(n,dtype=float)
        feat[idx]=1
        return feat