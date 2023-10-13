{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "M0H1XvWgB9ck"
      },
      "outputs": [],
      "source": [
        "# Install Stanza library from Stanford University\n",
        "!pip install stanza\n",
        "# Download the required files for using Stanza\n",
        "stanza.download('en')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rXKLW5VKDMTm"
      },
      "outputs": [],
      "source": [
        "# Import NLTK library for text processing\n",
        "import nltk\n",
        "from nltk import word_tokenize\n",
        "from nltk import sent_tokenize\n",
        "nltk.download('all')\n",
        "\n",
        "# Import Stanza for text processing\n",
        "import stanza\n",
        "\n",
        "# Suppress any warnings that may come up during code execution\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rb5w5yiFHs1b"
      },
      "outputs": [],
      "source": [
        "# You can change this to any other set of sentences\n",
        "doc = \"Oppenheimer was the leader of the Manhattan project. Oppenheimer was directed by Christopher Nolan.\""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tnMy-09oQyCK"
      },
      "source": [
        "# NLTK Functions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "olqoUP1-IVqx"
      },
      "outputs": [],
      "source": [
        "# Tokenize the document into separate sentences\n",
        "# The output is a list of sentences\n",
        "sent_tokenize(doc)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SLlykELYIH6r"
      },
      "outputs": [],
      "source": [
        "# Tokenize the document into separate words\n",
        "# The output is a list of words and punctuations\n",
        "word_tokenize(doc)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9zda7RnUPcsx"
      },
      "outputs": [],
      "source": [
        "# Part of Speech (POS) Tagging\n",
        "nltk.pos_tag(word_tokenize(doc))\n",
        "# Tag Meanings : https://www.guru99.com/pos-tagging-chunking-nltk.html"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6FBtsAo3Q1eE"
      },
      "source": [
        "# Stanford Stanza Functions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mvnKDREqQ4Ja"
      },
      "outputs": [],
      "source": [
        "# Documentation : https://stanfordnlp.github.io/stanza/index.html\n",
        "\n",
        "# Initialise the Stanza Pipepine\n",
        "nlp = stanza.Pipeline('en')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dIqElaC3Q4Ma"
      },
      "outputs": [],
      "source": [
        "# Process the previously defined text and store the output in stanza_doc\n",
        "stanza_doc = nlp(doc)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "92E8YLvtQ4PS"
      },
      "outputs": [],
      "source": [
        "# Print each individual sentence in the text\n",
        "for sentence in stanza_doc.sentences:\n",
        "    print(sentence.text)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5aLBm-MXSgMx"
      },
      "outputs": [],
      "source": [
        "# Convert the output to a list of dictionary\n",
        "len(stanza_doc.to_dict())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yNVVh_NESMct"
      },
      "outputs": [],
      "source": [
        "# Output the Named Entities for the text\n",
        "stanza_doc.entities"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aSNTovnbQ4R8"
      },
      "outputs": [],
      "source": [
        "# Print the detailed output for each word in the text\n",
        "for sentence in stanza_doc.sentences:\n",
        "    for item in sentence.to_dict():\n",
        "        print(item)\n",
        "    print(\"\\n\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vShCsjMjYL0X"
      },
      "source": [
        "# Some Applications of Classical NLP"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "llvLtCJtaxPO"
      },
      "source": [
        "[Text Classification](https://medium.com/@atmabodha/fictometer-a-simple-and-explainable-algorithm-for-sentiment-analysis-31186d2a8c7e)\n",
        "\n",
        "[Spam Filtering](https://towardsdatascience.com/how-to-identify-spam-using-natural-language-processing-nlp-af91f4170113)\n",
        "\n",
        "[Machine Translation](https://www.scaler.com/topics/machine-translation-in-nlp/)\n",
        "\n",
        "[Information Retrieval](https://www.tutorialspoint.com/natural_language_processing/natural_language_processing_information_retrieval.htm)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ij9O0Aqtko6V"
      },
      "source": [
        "# Word2Vec Embedding"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "UeIluqg8krKX"
      },
      "outputs": [],
      "source": [
        "%%time\n",
        "import gensim.downloader as api\n",
        "w2v = api.load('word2vec-google-news-300')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "w_trWO2glXtj"
      },
      "outputs": [],
      "source": [
        "w2v['language']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zhZPcMwYlpRi"
      },
      "outputs": [],
      "source": [
        "word = \"XYZ123\"\n",
        "\n",
        "try:\n",
        "    vec_word = w2v[word]\n",
        "except:\n",
        "    print(\"The word \" + word + \" does not appear in this model\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NffEsmUmmAVO"
      },
      "outputs": [],
      "source": [
        "word1 = \"King\"\n",
        "word2 = \"Queen\"\n",
        "\n",
        "w2v.similarity(word1,word2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_VE7IiBSosF2"
      },
      "outputs": [],
      "source": [
        "word1 = \"planet\"\n",
        "word2 = \"earth\"\n",
        "\n",
        "w2v.similarity(word1,word2)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2gpj19zqkjGD"
      },
      "source": [
        "# ELMo : Embeddings for Languade Models"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VSWZg5-LkkvF"
      },
      "outputs": [],
      "source": [
        "import tensorflow_hub as hub\n",
        "import tensorflow.compat.v1 as tf\n",
        "tf.disable_eager_execution()\n",
        "\n",
        "# Load pre trained ELMo model\n",
        "elmo = hub.Module(\"https://tfhub.dev/google/elmo/3\", trainable=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "biucx19Nr0LU"
      },
      "outputs": [],
      "source": [
        "init = tf.initialize_all_variables()\n",
        "sess = tf.Session()\n",
        "sess.run(init)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AUabCuHvqcZT"
      },
      "outputs": [],
      "source": [
        "# create an instance of ELMo\n",
        "embeddings = elmo(\n",
        "    [\n",
        "        \"I love to watch TV\",\n",
        "        \"I am wearing a wrist watch\",\n",
        "        \"My brother goes to the ground every Sunday to watch Football\",\n",
        "        \"My wife gifted me a beautiful watch on my birthday\"\n",
        "    ],\n",
        "    signature=\"default\",\n",
        "    as_dict=True)[\"elmo\"]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "z9Xk1cVZqt_Y"
      },
      "outputs": [],
      "source": [
        "# Compute the ELMo Embeddings\n",
        "word1_embedding = sess.run(embeddings[0][3])\n",
        "\n",
        "word2_embedding = sess.run(embeddings[1][5])\n",
        "\n",
        "word3_embedding = sess.run(embeddings[2][9])\n",
        "\n",
        "word4_embedding = sess.run(embeddings[3][6])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "THxvIqoFqyMI"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics.pairwise import cosine_similarity"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZRoUIBxDqvV3"
      },
      "outputs": [],
      "source": [
        "cosine_similarity([word1_embedding], [word2_embedding])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HO7OtFS_q_py"
      },
      "outputs": [],
      "source": [
        "cosine_similarity([word1_embedding], [word3_embedding])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4gULJWtPsExc"
      },
      "outputs": [],
      "source": [
        "cosine_similarity([word2_embedding], [word4_embedding])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jdzF7i38sHPJ"
      },
      "outputs": [],
      "source": [
        "cosine_similarity([word2_embedding], [word3_embedding])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6SeayRvEsImm"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}