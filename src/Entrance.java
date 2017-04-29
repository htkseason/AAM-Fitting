
import java.io.IOException;
import java.util.UUID;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import pers.season.vml.statistics.appearance.AppearanceFitting;
import pers.season.vml.statistics.appearance.AppearanceModel;
import pers.season.vml.statistics.appearance.AppearanceModelTrain;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.shape.ShapeModelTrain;
import pers.season.vml.statistics.texture.TextureInstance;
import pers.season.vml.statistics.texture.TextureModel;
import pers.season.vml.statistics.texture.TextureModelTrain;
import pers.season.vml.util.*;

public final class Entrance {

	static {
		// todo: x64/x86 judge
		// System.loadLibrary("lib/opencv_java2413_x64");
		System.loadLibrary("lib/opencv_java320_x64");
	}

	public static void main(String[] args) throws IOException {

		// RegressorTrain.train();
		aamFittingDemo();
		// RegressorTrain.trainLineR();

		System.out.println("program ended.");

	}

	public static void aamFittingDemo() {
		System.out.println("parallel threads : " + Runtime.getRuntime().availableProcessors());
		FaceDetector fd = FaceDetector.load("models/lbpcascade_frontalface.xml");
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", MuctData.default_ignore);

		// ShapeModelTrain.train("models/shape/", 0.90, false);
		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		// TextureModelTrain.train("models/texture/", 0.98, 20, 30, false);
		TextureModel tm = TextureModel.load("models/texture/", "U", "X_mean", "Z_e", "meanShape", "delaunay");

		// AppearanceModelTrain.train(sm, tm, "models/appearance/", 1.5, 0.98,
		// false);
		AppearanceModel am = AppearanceModel.load(sm, tm, "models/appearance/", "U", "Z_e", "shapeWeight");

		ImUtils.startTiming();
		// Mat pic = MuctData.getGrayJpg(0);

		Mat pic = Imgcodecs.imread("test.jpg", Imgcodecs.IMREAD_GRAYSCALE);

		Rect faceRect = fd.searchFace(pic).get(0);

		ImUtils.imshow(pic);
		pic.convertTo(pic, CvType.CV_32F);
		Mat v_pic = new Mat(pic.size(), pic.type());

		TextureInstance texture = new TextureInstance(tm);
		// texture.setFromPic(pic, MuctData.getPtsMat(0));
		ShapeInstance shape = new ShapeInstance(sm);

		shape.setFromParams(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
				faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);

		AppearanceFitting app = new AppearanceFitting(am, pic);
		app.setFromModels(shape.getZ(), texture.getZ());

		double preCost = Double.MAX_VALUE;
		JFrame win = new JFrame();

		for (int iter = 0; iter < 1000; iter++) {
			pic.copyTo(v_pic);
			app.printTo(v_pic, false);
			shape.setFromPts(sm.getXfromZ(app.getShapeZ()));
			Rect r = shape.getLocation();
			Imgproc.rectangle(v_pic, r.tl(), r.br(), new Scalar(255), 2);

			System.out.println("iter=" + iter + "\t\tcost=" + (int) ImUtils.getCostE(app.getCost()) + "\t\ttime="
					+ (int) ImUtils.getTiming() + " ms");
			ImUtils.imshow(win, v_pic, 1);

			ImUtils.startTiming();
			System.gc();
			Mat gra = app.getGradient();

			double descentRate = 1;
			while (descentRate > 0.05) {
				Mat step = new Mat();
				Core.multiply(gra, new Scalar(descentRate), step);
				app.updata(step);
				double cost = ImUtils.getCostE(app.getCost());
				if (preCost < cost) {
					Core.multiply(step, new Scalar(-1), step);
					app.updata(step);
					descentRate /= 2;
				} else {
					preCost = cost;
				}
			}

			Mat appShapeZ = app.getShapeZ();
			Mat appTextureZ = app.getTextureZ();
			app.getAppearanceModel().getShapeModel().clamp(appShapeZ, 3);
			app.setFromModels(appShapeZ, appTextureZ);

		}

		System.out.println("\ndone!");

	}

}
