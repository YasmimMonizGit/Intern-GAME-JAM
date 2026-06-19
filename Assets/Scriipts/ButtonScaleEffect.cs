using UnityEngine;

public class ButtonScaleEffect : MonoBehaviour
{
    public Transform visualTarget;
    public Vector3 pressedScale = new Vector3(1.2f, 1.2f, 1.2f);
    public float duration = 0.15f;

    private Vector3 normalScale;

    void Awake()
    {
        normalScale = visualTarget.localScale;
    }

    public void PressButton()
    {
        StopAllCoroutines();
        StartCoroutine(ScaleEffect());
    }

    private System.Collections.IEnumerator ScaleEffect()
    {
        visualTarget.localScale = pressedScale;

        yield return new WaitForSecondsRealtime(duration);

        visualTarget.localScale = normalScale;
    }
}