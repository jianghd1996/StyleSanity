using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using UnityEngine;
using System.IO;

[RequireComponent (typeof(ImageSynthesis))]
public class ExampleUI : MonoBehaviour {

	private int imageCounter = 1;

	void OnGUI ()
	{
		if (GUILayout.Button("Captcha!!! (" + imageCounter + ")"))
		{
			var sceneName = SceneManager.GetActiveScene().name;

			string path = "Screen/" ;
			ImageSynthesis imgs = GetComponent<ImageSynthesis>();
			// NOTE: due to per-camera / per-object motion being calculated late in the frame and after Update()
			// capturing is moved into LateUpdate (see ImageSynthesis.cs Known Issues)
			imgs.Save(sceneName + "_" + imageCounter++, imgs.width, imgs.height, path);
		}
	}
}
