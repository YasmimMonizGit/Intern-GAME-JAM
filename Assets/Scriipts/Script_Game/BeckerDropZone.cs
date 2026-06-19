using UnityEngine;

public class BeckerDropZone : MonoBehaviour
{
    public RecipeManager recipeManager;
    public RectTransform dropArea;
    public Transform beckerPoint;

    public bool IsInsideBecker(RectTransform item)
    {
        return RectTransformUtility.RectangleContainsScreenPoint(
            dropArea,
            item.position
        );
    }

    public bool TryDrop(GameObject item)
    {
        RectTransform itemRect = item.GetComponent<RectTransform>();

        if (IsInsideBecker(itemRect))
        {
            recipeManager.AddItem(item);
            Debug.Log("Item accepted");
            return true;
        }

        Debug.Log("Item outside becker");
        return false;
    }
}