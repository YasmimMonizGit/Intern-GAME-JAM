using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class RecipeManager : MonoBehaviour
{
    [Header("Recipe Setup")]
    public RecipeStep[] steps;

    [Header("UI")]
    public TextMeshProUGUI instructionText;

    [Header("Feedback")]
    public string successText = "Perfect mixture!";
    public string failText = "Wrong formula!";

    [Header("Becker Liquid")]
    public Image beckerLiquid;

    [Header("Audio")]
    public AudioSource dropSound;
    public AudioClip wrongCombinationClip;

    private List<GameObject> playerSequence = new List<GameObject>();
    private int currentStep = 0;
    private Color originalColor;

    public GameObject nextLevelButton;

    void Start()
    {
        originalColor = beckerLiquid.color;
        UpdateInstruction();
    }

    void UpdateInstruction()
    {
        if (currentStep < steps.Length)
        {
            instructionText.text = steps[currentStep].instructionText;
        }
        else
        {
            CheckFinalSequence();
        }
    }

    public void AddItem(GameObject item)
    {
        playerSequence.Add(item);
        if (dropSound != null)
        {
            dropSound.Play();
        }

        ChangeLiquidColor();

        currentStep++;
        UpdateInstruction();
    }

    void ChangeLiquidColor()
    {
        Color randomColor = new Color(
            Random.value,
            Random.value,
            Random.value
        );

        beckerLiquid.color = randomColor;
    }

    void CheckFinalSequence()
    {
        int mistakesThisRound = 0;

        for (int i = 0; i < steps.Length; i++)
        {
            if (playerSequence[i] != steps[i].correctItem)
            {
                mistakesThisRound++;
            }
        }

        if (mistakesThisRound == 0)
        {
            instructionText.text = successText;

            if (nextLevelButton != null)
            {
                nextLevelButton.SetActive(true);
            }

            Debug.Log("SUCCESS");
        }
        else
        {
            for (int i = 0; i < mistakesThisRound; i++)
            {
                GameManager.Instance.AddMistake();
            }

            StartCoroutine(FailSequence());
        }
    }

    IEnumerator FailSequence()
    {
        instructionText.text = failText;

        float timer = 0f;

        if (dropSound != null && wrongCombinationClip != null)
        {
            dropSound.PlayOneShot(wrongCombinationClip);
        }


        while (timer < 2f)
        {
            

            ChangeLiquidColor();

            yield return new WaitForSeconds(0.5f);

            timer += 0.5f;
        }
       
        ResetRecipe();
    }

    void ResetRecipe()
    {
        playerSequence.Clear();
        currentStep = 0;

        beckerLiquid.color = originalColor;

        UpdateInstruction();

        Debug.Log("Recipe Reset");
    }
}