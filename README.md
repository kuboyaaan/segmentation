# segmentation
<p>物体検知(segmentationタスク)に対して、処理を行った際のコードです．<br>
もともとJupyter Notebook(.ipynb)だったコードをメソッドごとに.pyファイルに変更しています．逆に見づらくなっていたら申し訳ありません．</p>
<h2>構成</h2>
<h3>main.py</h3>
物体検知を行った際の処理を記述．画像分類の処理は未記述．
<h3>coder.py</h3>
<p>ラベルとなるsegment領域情報が位置のピクセル値で与えられるため、そのデータからmask画像を作成するdecoderと逆に推定したmask画像から位置のピクセル値を作成するencoderを記述</p>
<h3>loss.py</h3>
<p>物体検知では、評価指標にIoUを用いていたため、IoUの損失関数を記述．</p>
<a href="https://mathwords.net/iou>IoU(評価指標)の意味と値の厳しさ-具体例で学ぶ数学"</a>
<h3>models.py</h3>
<p>物体検知モデル(U-Net)と画像分類モデル(ResNetのfine-tuning)を記述．<br>
  物体検知のみ、fit, show_loss, prediction関数を記述</p>
<h3>generator.py</h3>
<p>datageneratorを記述(物体検知では、make_image_gen関数のみ使用)</p>
<p>その他のメソッド：画像分類を行った際、画像サイズ指定して読み込み学習を行うと、縦横比がバラバラなデータは間延びしたような（ぼやけた）画像での学習になってしまうという問題があったため、縦横比を保ち、周りを0でpaddingして正方形の画像を生成する必要があったため作成．</p>
