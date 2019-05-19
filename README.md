# segmentation
#構成
<h1>main.py</h1>
物体検知を行った際の処理を記述．画像分類の処理は未記述．
<h1>coder.py</h1>
<p>ラベルとなるsegment領域情報が位置のピクセル値で与えられるため、そのデータからmask画像を作成するdecoderと逆に推定したmask画像から位置のピクセル値を作成するencoderを記述</p>
<h1>loss.py</h1>
<p>物体検知では、評価指標にIoUを用いていたため、IoUの損失関数を記述．</p>
<a href="https://mathwords.net/iou>IoU(評価指標)の意味と値の厳しさ-具体例で学ぶ数学"</a>
<h1>models.py</h1>
<p>物体検知モデル(U-Net)と画像分類モデル(ResNetのfine-tuning)を記述．<br>
  物体検知のみ、fit, show_loss, prediction関数を記述</p>
<h1>generator.py</h1>
<p>画像分類を行った際、画像サイズ指定して読み込み学習を行うと、縦横比がバラバラなデータは間延びしたような（ぼやけた）画像での学習になってしまうという問題があったため、縦横比を保ち、周りを0でpaddingして正方形の画像を生成する必要があったため作成．物体検知では用いていない．</p>
