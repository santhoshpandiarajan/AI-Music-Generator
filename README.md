# AI-Music-Generator

# Unleash the Melodic Magic


## Experience Music Beyond Imagination with AI Music Generator! 

Music has always been a captivating art form that resonates with people across cultures. In recent years, advancements in artificial intelligence and deep learning have opened up new possibilities for generating music using neural networks. In this blog, we will explore the fascinating process of generating music using LSTM (Long Short-Term Memory) neural networks. Our focus will be on leveraging the MusicNet dataset, a comprehensive collection of classical music MIDI files. Additionally, we will discuss the instrumental role played by Intel DevCloud in facilitating efficient code execution, enabling faster training, and leveraging the power of Intel hardware resources.

To begin our musical journey, we need to import several essential libraries. These include pandas and numpy for data manipulation, mido for handling MIDI files, IPython for audio playback, and matplotlib and librosa for visualizing audio data. Furthermore, we rely on Keras, a popular deep learning framework, to construct and train our LSTM model. These libraries equip us with the necessary tools to process MIDI files, extract musical information, and implement our neural network.

![1](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/8effdad8-5d02-4d78-8673-b772446c52f1)



The dictionary `key_notes` contains entries for various note names, including both natural notes (e.g., 'C', 'D', 'E') and sharps/flats (e.g., 'C#', 'Eb', 'Gb'). Each entry consists of a note name as the key and the corresponding MIDI note number as the value.

In MIDI representation, each note is assigned a unique number to represent its pitch. The dictionary allows us to easily convert between note names (e.g., 'C', 'D#', 'Bb') and their corresponding MIDI note numbers (e.g., 60, 63, 70).



![2](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/a696f3b9-825f-4b1a-be97-a5d70e48ee3f)



Before diving into the code, we start by loading the MusicNet dataset metadata using pandas. The metadata provides valuable information about the available MIDI files, such as composer, title, and duration. Familiarizing ourselves with the structure and content of the dataset allows us to navigate through the vast collection of MIDI files and select specific compositions for music generation.



![3](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/09e9fd08-e229-4af0-a399-d2455d6381ae)



The code begins by loading a MIDI file named `'2313_qt15_1.mid'` from the Beethoven folder in the MusicNet dataset. The `MidiFile` function from the `mido` library is used for this purpose. The `clip=True` argument ensures that any messages with timestamps outside the MIDI file's boundaries are truncated.

Next, we access the `tracks` attribute of the `mid` object. MIDI files consist of one or more tracks, which contain musical events such as notes, tempo changes, and control messages. By accessing `mid.tracks`, we obtain a list of all the tracks present in the MIDI file.



![4](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/b31c9cd9-3fe1-48f5-a3bf-d49acb6005e7)



The code then iterates over the second track (`mid.tracks[1]`) using a `for` loop. It checks if each item in the track is a metadata message by examining its string representation. If the string contains the word 'meta', it indicates that the message is a metadata message. In this case, the message is printed.



![5](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/3ec28417-93b2-47c5-90e9-2e1d2feb95a6)



Similarly, the code proceeds to the third track (`mid.tracks[2]`) and iterates over its items using another `for` loop. It prints each item and increments a counter variable `k`. The loop breaks when `k` becomes greater than 50, limiting the number of items printed.



![6](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/498ef834-d597-4c7b-833e-f474cef4fd8a)



This code allows us to inspect the metadata and contents of specific tracks in a MIDI file. It can be helpful for understanding the structure and content of MIDI files and extracting relevant musical information for further analysis or processing.

In this section of the code, we are working with MIDI files from the Beethoven folder of the MusicNet dataset. The code initializes an empty dictionary called `beethoven_midi_traks` to store the MIDI tracks for Beethoven's compositions. 

The variable `n` is set to 20, indicating that we want to process the first 20 MIDI files in the Beethoven folder. 

Inside the loop, for each value of `m` ranging from 0 to 19, the code loads a MIDI file using the `MidiFile` function from the `mido` library. The file path is constructed by concatenating the Beethoven folder path with the MIDI file name obtained from the list of files in the Beethoven folder.

The `clip=True` argument ensures that any messages with timestamps outside the MIDI file's boundaries are truncated.

After loading the MIDI file, the code prints the `tracks` attribute, which gives us a list of all the tracks present in the MIDI file.

Next, another loop is used to iterate over the tracks of the MIDI file. For the first track (index 0), the name of the track is stored in the `name` variable with a colon appended. This name represents the composition or movement associated with the MIDI file.

For the subsequent tracks, their names are appended to the `name` variable, forming a key in the `beethoven_midi_traks` dictionary. The value of the key is set to the corresponding MIDI track object.

By the end of the loop, the `beethoven_midi_traks` dictionary contains entries for each MIDI track, organized by composition or movement. Each entry consists of a key in the format 'composition: track_name' and the corresponding MIDI track object.

This code allows us to organize and access the MIDI tracks of Beethoven's compositions, providing a convenient way to analyze and process individual tracks for further tasks such as note extraction or music generation.



![7](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/6a9f87ea-9b9e-4e29-ba9c-4fc4324dc6b2)



The code here defines a function called `get_key` that extracts the key signature from a string representation of a MIDI message. The function takes a string `s` as input and initializes a variable `k` as `None`. 

It checks if the string contains the word 'key'. If it does, the function proceeds to extract the key signature by slicing the string from index 33 to 35. 

Next, it checks if the last character of the extracted key signature is either 'm' or "'" (indicating a minor key). If it is, the last character is removed from the key signature by slicing it with `k[:-1]`. Finally, the function returns the extracted key signature.



![8](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/b834d2f2-8a85-4a50-85ea-726f436c09a8)



The code here defines a function called `parse_notes` that takes a MIDI track as input and returns a list of dictionaries, where each dictionary represents a note with its duration and velocity.

The function initializes some variables: 

- `key` is set to the default key signature 'C'.

- `tunes` is an empty list to store the parsed notes.

- `new_tune` is an empty list to store the notes of the current composition or movement.

- `note_dict` is an empty dictionary that will hold the information of each note.

The function then iterates over the MIDI track, processing each MIDI message. For meta messages (identified by the `is_meta` attribute), the function extracts the key signature using the `get_key` function (assuming it was defined previously). If a new key signature is found, it updates the `key` variable accordingly. If there are notes already parsed in `new_tune`, it appends them to `tunes` and resets `new_tune` to an empty list.

For note messages (identified by the `note_on` or `note_off` types), the function checks if it is a note-on message with a non-zero velocity and positive duration. If so, it creates a new dictionary `note_dict` and assigns the time, note value, velocity, and channel to the corresponding keys in the dictionary.

If it is a note-off message or a note-on message with zero velocity, it checks if there is a note dictionary with information. If so, it assigns the pause (duration until the note is released), the current key signature, and appends the note dictionary to `new_tune`. Then it resets `note_dict` to an empty dictionary.

After iterating through all the MIDI messages, it appends any remaining notes in `new_tune` to `tunes`. Finally, the function returns the list of parsed notes, where each note is represented by a dictionary containing its time, note value, velocity, pause duration, and key signature.

This function is useful for parsing MIDI tracks and extracting relevant information about the notes, such as their duration, velocity, and key signature. The resulting data structure can be further processed or analyzed for various musical applications.



![9](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/42f3eb20-6c48-4cd5-874a-b4701cce5b16)



The code here defines a function called `tune_to_midi` that takes a list of notes as input and generates a MIDI file based on those notes. It uses the `MidiFile` and `MidiTrack` classes from the `mido` library.

The function first creates a new `MidiFile` object and a `MidiTrack` object. The track is then appended to the MIDI file. Next, the function iterates over each note in the input `tune`. It checks the `debug_mode` flag to determine how to create MIDI messages for each note.

If `debug_mode` is `True`, the function appends MIDI messages with fixed values for note-on and note-off events, specifically using a time of 64 for note-on and 128 for note-off. This is useful for debugging or testing purposes.

If `debug_mode` is `False`, the function retrieves the relevant information from the note dictionary. It extracts the note value, velocity, time, and pause duration from the note dictionary and uses them to create the corresponding MIDI messages. The `note_on` message includes the note value, velocity, and time, while the `note_off` message includes the note value and the pause duration. After iterating over all the notes, the function saves the MIDI file with the specified `midi_name` and appends the ".mid" extension.



![10](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/0143feed-320f-44ba-b5dd-59e97f09a2ae)



The code segment here involves the creation of a list called `tunes` and the extraction of musical data from the `beethoven_midi_traks` dictionary. It specifically focuses on the tracks that contain the word "Right" in their keys.

First, an empty list called `tunes` is initialized. Additionally, the maximum key value from the `key_notes` dictionary is assigned to the variable `max_key`. This value represents the highest note value in the music.

The code then iterates over the items in the `beethoven_midi_traks` dictionary using a `for` loop. Each item consists of a key-value pair, where the key represents the track name and the value represents the MIDI track itself.

Within the loop, it checks if the string "Right" is present in the track name (`k`). If the condition is satisfied, it proceeds to extract the musical data from the MIDI track by calling the `parse_notes` function, passing the MIDI track (`v`) as an argument. The `parse_notes` function is expected to return a list of dictionaries, each containing information about a note, its duration, and velocity.

Next, it checks if the length of the `new_tunes` list is greater than zero, indicating that at least one track was successfully parsed. If this condition is met, it appends the first element of the `new_tunes` list (accessed with index 0) to the `tunes` list as a pandas DataFrame. Finally, it retrieves the third element from the `tunes` list (accessed with index 2) and returns it.



![11](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/eda67f5d-5dd7-439c-aad4-bbbb4070d866)





![12](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/bd1a0609-6f82-4ab4-a71c-94b6bbb9ff1e)



The code segment here involves the preparation of the training data for a machine learning model. It includes the `various` function and the generation of the `X` and `y` arrays.

The `various` function takes a list of notes as input (`notes`). Its purpose is to determine whether the given sequence of notes exhibits sufficient variation. It checks if, within a sliding window of size 8, there are at least three unique notes. The function returns a boolean value indicating whether the variation condition is met.

Next, the code sets the `phrase_len` variable to a value of 60. This variable represents the length of a musical phrase. Two empty lists, `X` and `y`, are created to store the input and target data, respectively, for the machine learning model.

The code then iterates over the `tunes` list, which contains pandas DataFrames of parsed musical data. For each DataFrame (`t`) in `tunes`, an inner loop runs from 0 to the length of the DataFrame minus the `phrase_len` value. This loop generates sliding windows of length `phrase_len` over the DataFrame.

Within the inner loop, it checks if the current window of notes, from index `i` to `i + phrase_len`, satisfies the variation condition by calling the `various` function. If the condition is met, it appends the corresponding subset of the DataFrame (`t.iloc[i:i + phrase_len, :3]`) to the `X` list, representing the input sequence. Additionally, it appends the next row of the DataFrame (`t.iloc[i + phrase_len, :3]`) to the `y` list, representing the target note. Finally, the `X` and `y` lists are converted to NumPy arrays (`np.array(X)` and `np.array(y)`) for further processing.



![13](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/0694f97d-208b-49ed-80b7-889c1d4f2e72)





![14](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/f8676a96-fce7-4ad3-95ec-1a24850069a8)



After that, the `X` and `y` arrays are cast to the integer data type using the `astype(int)` method. This conversion ensures that the values in `X` and `y` are represented as integers. To retrieve the shape of the `X` array, we are using the `shape` attribute.



![15](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/ac4a0525-7349-4020-bb3f-ef7912fa8a56)



The code here shows the construction, compilation, and training of a sequential model using Keras. Here's an explanation of each step:

1. Model Creation:

   - `model = Sequential()`: Creates a sequential model, which is a linear stack of layers.

2. Adding Layers:

   - `model.add(LSTM(512, return_sequences=False, input_shape=(phrase_len, 3)))`: Adds an LSTM layer with 512 units to the model. The `return_sequences=False` argument indicates that this LSTM layer will only return the final output rather than the full sequence. The `input_shape` parameter specifies the shape of the input data, which is `(phrase_len, 3)`.

   - `model.add(Dropout(0.5))`: Adds a dropout layer with a dropout rate of 0.5. Dropout is a regularization technique that randomly sets a fraction of input units to 0 during training, which helps prevent overfitting.

   - `model.add(Dense(3, activation='relu'))`: Adds a fully connected dense layer with 3 units and ReLU activation. This layer will map the output from the LSTM layer to a 3-dimensional output.

3. Model Compilation:

   - `model.compile(loss='mae', optimizer='adam')`: Compiles the model by specifying the loss function and optimizer. Here, the loss function is Mean Absolute Error (`'mae'`), which measures the average absolute difference between the predicted and true values. The optimizer used is Adam, which is an adaptive learning rate optimization algorithm.

4. Model Training:

   - `model.fit(X, y, batch_size=256, epochs=100, validation_split=0.2)`: Trains the model using the provided input data `X` and target data `y`. The `batch_size` parameter determines the number of samples to be processed at once, and the `epochs` parameter specifies the number of training iterations. The `validation_split` parameter of 0.2 indicates that 20% of the data will be used for validation during training.

During the training process, the model learns to minimize the specified loss function by adjusting its weights and biases based on the provided input data and target values. The validation split helps monitor the model's performance on unseen data and prevents overfitting.



![16](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/458a934d-501c-46f0-a0c0-dfaf0ebf6aec)



The `tune_generator` function generates multiple tunes using the provided LSTM model. Each tune is saved as a MIDI file with a unique name.

1. The function `tune_generator` takes the trained LSTM model as input, along with an optional `name` parameter to specify the name prefix for the generated MIDI files.

2. A loop runs three times to generate three different tunes.

3. In each iteration, a random starting point (`start`) within the range of the available input data (`X`) is chosen.

4. The `pattern` is initialized with the selected starting point.

5. A loop runs for 100 iterations, generating predictions for the next note based on the previous pattern using the LSTM model. The `prediction_input` is reshaped to match the model's expected input shape, and the prediction is obtained using `model.predict`. The predicted note is appended to the `prediction_output` list.

6. After each prediction, the `pattern` is updated by appending the prediction and removing the first element. This allows the pattern to slide forward for the next iteration.

7. The `prediction_output` is converted into a DataFrame (`notes`) with columns for time, note, velocity, and pause. The `pause` column is set to a fixed value of 180, representing the duration of the pause between notes.

8. The DataFrame (`notes`) is converted to a list of dictionaries (`notes_dict`) using the `to_dict` method with the `'records'` parameter.

9. The `tune_to_midi` function is called with the generated notes as input, along with the MIDI file name constructed using the `name` parameter and the iteration index.



![17](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/c8043fdd-91ca-4b01-8d41-bed6265264b1)



This function plays a MIDI file using the music21 library. The `filename` parameter specifies the path or name of the MIDI file to be played. The function loads the MIDI file, converts it to a music21 stream object, and plays it back using the `show('midi')` method. It calls the `play_midi` function and passes the filename as an argument.



![18](https://github.com/santhoshpandiarajan/AI-Music-Generator/assets/131739054/d64578aa-4c2f-46d7-bf67-5427a92cbb94)


