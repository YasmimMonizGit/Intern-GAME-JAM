using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIColorChanger : MonoBehaviour
{
    public Image targetImage;
    public Color[] colors = new Color[5];
    public float changeTime = 0.5f;

    private int currentIndex = 0;

    private void Start()
    {
        StartCoroutine(ChangeColors());
    }

    IEnumerator ChangeColors()
    {
        while (true)
        {
            if (targetImage != null && colors.Length > 0)
            {
                targetImage.color = colors[currentIndex];

                currentIndex++;
                if (currentIndex >= colors.Length)
                {
                    currentIndex = 0;
                }
            }

            yield return new WaitForSeconds(changeTime);
        }
    }
}