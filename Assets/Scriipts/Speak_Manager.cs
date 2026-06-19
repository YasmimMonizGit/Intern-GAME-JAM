using UnityEngine;
using UnityEngine.SceneManagement;

public class Speak_manager : MonoBehaviour
{
    public GameObject[] elements; // elementos da sequência
    public string nextSceneName; // nome da cena no inspector

    private int currentIndex = 0;

    void Start()
    {
        // desliga todos
        for (int i = 0; i < elements.Length; i++)
        {
            elements[i].SetActive(false);
        }

        // ativa o primeiro
        if (elements.Length > 0)
        {
            elements[0].SetActive(true);
        }
    }

   

    public void NextElement()
    {
        if (elements.Length == 0) return;

        // desliga atual
        elements[currentIndex].SetActive(false);

        currentIndex++;

        // se ainda houver elementos
        if (currentIndex < elements.Length)
        {
            elements[currentIndex].SetActive(true);
        }
        else
        {
            FinalAction();
        }
    }

    public void FinalAction()
    {
        if (!string.IsNullOrEmpty(nextSceneName))
        {
            SceneManager.LoadScene(nextSceneName);
        }
    }
}