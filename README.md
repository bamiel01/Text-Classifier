
# Text Classifier


A Python script for training a basic classifier. The model is intended to infer the author of a piece of text, with the specific use case being a message in a group chat.

## Documentation

Data should be in a file called `data.csv` with the first header being `author` and the second being `text`. In table form, the CSV should have the following schema:

| author        | text        |
|---------------|-------------|
| Person's name | The message |
| Person's name | The message |
| Person's name | The message |
| Person's name | The message |


The output after training will be `config.json`, `pytorch_model.bin`, and `training_args.bin` in the `results` folder.



To get the data, you can use [this repo](https://github.com/bamiel01/Get-Messages) to extract the data from your computer (Mac only). The result will be `handle_id`s instead of names, which you can manually change (should be trivial: find, replace all feature).

