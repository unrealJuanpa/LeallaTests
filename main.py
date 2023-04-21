import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import numpy as np

# encoder = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-small/1")
# encoder = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-base/1")
encoder = hub.KerasLayer("https://tfhub.dev/google/LEALLA/LEALLA-large/1")

english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
italian_sentences = tf.constant(["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."])
japanese_sentences = tf.constant(["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"])
spanish_sentences = tf.constant(["hola", "saludos", "el perro es muy bonito"])

english_embeds = encoder(english_sentences)
japanese_embeds = encoder(japanese_sentences)
italian_embeds = encoder(italian_sentences)
spanish_embeds = encoder(spanish_sentences)

# English-Italian similarity
print(np.matmul(english_embeds, np.transpose(italian_embeds)))

# English-Japanese similarity
print(np.matmul(english_embeds, np.transpose(japanese_embeds)))

# Italian-Japanese similarity
print(np.matmul(italian_embeds, np.transpose(japanese_embeds)))

print("spanish")
print(spanish_embeds)
print(spanish_embeds[0].shape)