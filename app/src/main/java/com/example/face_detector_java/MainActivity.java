package com.example.face_detector_java;

import static androidx.camera.core.internal.utils.ImageUtil.rotateBitmap;
import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.YuvImage;
//import android.graphics.Rect;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
//import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


public class MainActivity extends AppCompatActivity {

    private ImageView frameView;
    private CascadeClassifier faceCascade;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        frameView = findViewById(R.id.frameView);

        if (OpenCVLoader.initDebug()) {
            Toast.makeText(this, "Open CV Instalado correctamente.", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this, "Open CV no detectado.", Toast.LENGTH_LONG).show();
        }
        // Solicitar permiso de la cámara si no está concedido
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 100);
        } else {
            startCamera(); // Inicia la cámara si el permiso ya está concedido
        }
        loadCascade();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == 100 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            // Muestra un mensaje indicando que el permiso es necesario
            Toast.makeText(this, "Permiso de cámara denegado. La funcionalidad no estará disponible.", Toast.LENGTH_LONG).show();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // Configurar ImageAnalysis
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // Solo procesa el frame más reciente
                        .build();

                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), this::analyzeFrame);

                // Seleccionar la cámara trasera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();

                // Vincular el ImageAnalysis al ciclo de vida
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyzeFrame(@NonNull ImageProxy image) {

        Bitmap bitmap = imageProxyToBitmap(image);

        Mat mat = bitmapToMat(bitmap);             // Convierte Bitmap a Mat de OpenCV

        detectFaces(mat);  // Aplica detección de rostros

        Bitmap finalImage = matToBitmap(mat);  // Convierte el Mat procesado a Bitmap

        runOnUiThread(() -> frameView.setImageBitmap(finalImage));

        // Cierra el frame para liberar memoria
        image.close();
    }

    // Convierte Bitmap a Mat
    private Mat bitmapToMat(Bitmap bitmap) {
        Mat mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }

    // Convierte Mat a Bitmap
    private Bitmap matToBitmap(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }

    private void detectFaces(Mat frame) {
        if (faceCascade == null) {
            Toast.makeText(this, "CascadeClassifier no cargado.", Toast.LENGTH_LONG).show();
            return;
        }
        Core.flip(frame, frame, 1);
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY);

        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, new Size(40, 40), new Size());

        for (org.opencv.core.Rect rect : faces.toArray()) {
            Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
        }
    }

    private void loadCascade() {
        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            faceCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer(); // Plano Y
        ByteBuffer uBuffer = planes[1].getBuffer(); // Plano U
        ByteBuffer vBuffer = planes[2].getBuffer(); // Plano V

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        // Crear un array NV21
        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        // Convertir NV21 a Bitmap
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
        byte[] jpegBytes = out.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);

        // Corregir la rotación según la orientación de la cámara
        return rotateBitmap(bitmap, image.getImageInfo().getRotationDegrees());
    }
}

