// Importar las librerías necesarias
const faceapi = require('@vladmandic/face-api');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const sharp = require('sharp'); // <--- Importar sharp

// URL donde están alojados los modelos de face-api
const MODEL_URL = 'https://cubillosandrey.github.io/reconocimiento-models/';

// Variable para asegurar que los modelos se carguen una sola vez
let modelsLoaded = false;

// Función para cargar los modelos
async function loadModels() {
  if (!modelsLoaded) {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
    ]);
    modelsLoaded = true;
    console.log('Modelos cargados');
  }
}

// NUEVA VERSIÓN de la función para obtener el descriptor
async function getDescriptorFromUrl(imageUrl) {
  try {
    // Descargar la imagen
    const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
    const imageBuffer = Buffer.from(response.data);

    // Redimensionar la imagen para ahorrar memoria
    const resizedImageBuffer = await sharp(imageBuffer)
      .resize({ width: 600 }) // Redimensiona al ancho de 600px, manteniendo la proporción
      .toBuffer();

    // Decodificar el buffer redimensionado a un tensor
    const image = tf.node.decodeImage(resizedImageBuffer);

    // Detectar la cara y obtener el descriptor
    const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
    tf.dispose(image); // Liberar memoria del tensor
    return detection ? detection.descriptor : null;
  } catch (error) {
    console.error(`Error procesando la imagen ${imageUrl}:`, error);
    return null;
  }
}


// Handler principal del API (sin cambios)
module.exports = async (req, res) => {
  try {
    // Cargar los modelos si no están cargados
    await loadModels();

    // Obtener las URLs de los parámetros de la consulta (p1 y p2)
    const { p1, p2 } = req.query;

    if (!p1 || !p2) {
      return res.status(400).json({ error: 'Faltan los parámetros de URL p1 y p2.' });
    }

    // Obtener los descriptores de ambas imágenes en paralelo
    const [descriptor1, descriptor2] = await Promise.all([
      getDescriptorFromUrl(p1),
      getDescriptorFromUrl(p2)
    ]);
    
    if (!descriptor1 || !descriptor2) {
      return res.status(200).json({ resultado: '⚠️ No se detectó un rostro en una o ambas imágenes.' });
    }

    // Calcular la distancia y comparar
    const distancia = faceapi.euclideanDistance(descriptor1, descriptor2);
    const umbral = 0.6;
    const sonLaMismaPersona = distancia < umbral;
    
    let resultadoFinal;
    if (sonLaMismaPersona) {
      resultadoFinal = `✅ Son la misma persona (Distancia: ${distancia.toFixed(4)})`;
    } else {
      resultadoFinal = `❌ No son la misma persona (Distancia: ${distancia.toFixed(4)})`;
    }

    // Devolver el resultado en formato JSON
    return res.status(200).json({ resultado: resultadoFinal });

  } catch (error) {
    console.error('Error en el handler principal:', error);
    return res.status(500).json({ error: 'Ocurrió un error interno en el servidor.' });
  }
};
