# Signal Extraction and Price Forecasting with LSTM and k-means

LSTM Composite implementation in TensorFlow described in [1] by combining an autoencoder and a Seq2Seq for its predictor<br /> Extracted n-day market pattern using k-means from Scikit-Learn

## Getting Started
The following are the required packages to run the models. 

### Installing
We recommend setting up a virtual environment with Python > 2.7.x (tested on 2.7.10):
```
virtualenv -p python venv
source venv/bin/activate
```
Install all required packages by running:
```
pip install -r requirements.txt
```
Run the notebooks with `jupyter notebook`
### Saved Models
In addition you can use this [checkpoint](https://www.dropbox.com/s/dcxktu8bsvwuxga/btcusd-ckpts.zip?dl=0) to initialize your model for the demo (more checkpoints to come). <br />
The model is trained on ```load_OHLC_no_vol()``` with ```hidden_size=[128], encoder_steps=24, decoder_steps=24```.
## Built With

* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

* [1] https://arxiv.org/pdf/1502.04681.pdf
