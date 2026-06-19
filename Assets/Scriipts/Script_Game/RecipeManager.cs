using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class RecipeManager : MonoBehaviour
{
    public RecipeStep[] steps;
    public TextMeshProUGUI instructionText;

    public string successText = "Perfect mixture!";
    public string failText = "Wrong formula!";

    private List<GameObject> playerSequence = new List<GameObject>();
    private int currentStep = 0;

    void Start()
    {
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
        currentStep++;
        UpdateInstruction();
    }

    void CheckFinalSequence()
    {
        bool correct = true;

        for (int i = 0; i < steps.Length; i++)
        {
            if (playerSequence[i] != steps[i].correctItem)
            {
                correct = false;
                break;
            }
        }

        if (correct)
        {
            instructionText.text = successText;
            Debug.Log("SUCCESS");
        }
        else
        {
            instructionText.text = failText;
            Debug.Log("FAIL");
        }
    }
}