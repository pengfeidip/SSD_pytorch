--- pengfei --- 2019年1月5日17:39:58

对  https://github.com/amdegroot/ssd.pytorch 的一些改造。使得在0.4.1版本的pytorch 也能获得同样的效果。

目前 `train.py` 文件留作备份，以`pf_main.py` 作为train的主文件



**update**

2019年1月9日15:48:03

引入了 `keep_top_k`参数，用以控制每张图最多的有效bounding box数目（default 200）

说明：

1.  `keep_top_k` 这个参数用于控制每张图最多的bounding box的数目。 因为原作者的程序有点问题，统一了 `top_k`和`keep_top_k`参数（见论文作者代码），然后在程序上并没有限制每张图最多的bounding box的数目（也可能在他的所用的pytorch版本上有这效果，但是0.4.1上没有）。
2. 实际上引入这个参数以后并没有什么变化![img](file:///C:\Users\ADMINI~1\AppData\Local\Temp\SGPicFaceTpBq\18408\577C8566.png).....，因为在经过`conf_thresh`以及`NMS`以后其实剩下的有效框已经不多了~
3. 真不知道我费这个劲干嘛...	









