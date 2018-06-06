# *Identifying Markers for Human Emotion in Breath Using Convolutional Autoencoders on Movie Data*

- ### data
  - **audio**: contains all the audio data for the 6 movies, as numpy matrices
  - **labels**: contains all the labels for the 6 movies, in the original CSV format and transposed
  - **pixels**: contains all the video data for the 12 movies the autoencoder is trained over, as numpy matrices
  - **vocs**: contains all the vocs for all 46 screenings of the 6 movies, in ARFF format

- ### models
  - **``labels_audio``**: classification network on audio data, trained over 1000 epochs
  - **``video_autoencoder``**: trained over 50 epochs
  - **``video_autoencoder_with_bn``**: trained over 50 epochs with batch normalization
  - **``vocs_audio_oso``**: prediction network on audio data, leaving one screening out for validation, trained over 1000 epochs
  - **``vocs_audio_l20``**: prediction network on audio data, using the last 20% of each movie for validation, trained over 1000 epochs
  
### [Download the original paper](https://github.com/TomFevrier/smelloffear-deeplearning/raw/master/Identifying%20Markers%20for%20Human%20Emotion%20in%20Breath%20Using%20Convolutional%20Autoencoders%20on%20Movie%20Data.pdf)

- ### ``convert_vae_to_nn.py``

  Used to create either a classification network or a prediction network from an autoencoder.

  Usage: ``python(3) convert_vae_to_nn.py autoencoder_version model_version output [-y y_test_structure] [-bn]``
  - ``output``: should be either ``labels`` or ``vocs``
  - ``-y y_test_structure``: ``oso``: One Screening Out or ``omo``: One Movie Out or ``l20``: Last 20% (if ``vocs``)
  - ``-bn``: if batch normalization was enabled

- ### ``extract_audio.py``
  Used to create a numpy matrix from the audio data of a movie. The sub_path containing the files needs to be set first.

  Usage: ``python(3) extract_audio.py film``


- ### ``extract_pixels.py``
  Used to create a numpy matrix from the video data of a movie. The sub_path containing the files needs to be set first.

  Usage: ``python(3) extract_pixels.py film nb_frames offset``
  - ``offset``: frame number to start with (in case the first few frames are black)

- ### ``labels_audio.py``
  Used to train a classification network on audio data. If the model does not exist, it is created.

  Usage: ``python(3) labels_audio.py model_version [-b batch_size] [-e epochs]``
  - ``-b batch_size``: default is 64
  - ``-e epochs``: default is 100

- ### ``labels_video.py``
  Used to train a classification network on video data. The model needs to be created from an autoencoder.

  Usage: ``python(3) labels_video.py model_version [-b batch_size] [-e epochs]``
  - ``-b batch_size``: default is 16
  - ``-e epochs``: default is 10

- ### ``video_autoencoder.py``
  Used to train a video autoencoder. If the model does not exist, it is created.

  Usage: ``python(3) video_autoencoder.py model_version [-b batch_size] [-e epochs] [-bn]``
  - ``-b batch_size``: default is 16
  - ``-e epochs``: default is 10
  - ``-bn``: enables batch normalization

- ### ``vocs_audio.py``
  Used to train a prediction network on audio data. If the model does not exist, it is created.

  Usage: ``python(3) vocs_audio.py model_version y_test_structure [-b batch_size] [-e epochs] [-f film_tested]``
  - ``y_test_structure``: ``oso``: One Screening Out or ``omo``: One Movie Out or ``l20``: Last 20%
  - ``-b batch_size``: default is 64
  - ``-e epochs``: default is 100
  - ``-f film_tested``: film left out (if ``omo``)

- ### ``vocs_video.py``
  Used to train a prediction network on video data. The model needs to be created from an autoencoder.

  Usage: ``python(3) vocs_video.py model_version y_test_structure [-b batch_size] [-e epochs] [-f film_tested]``
  - ``y_test_structure``: oso: One Screening Out or omo: One Movie Out or l20: Last 20%
  - ``-b batch_size``: default is 16
  - ``-e epochs``: default is 10
  - ``-f film_tested``: film left out (if ``omo``)
