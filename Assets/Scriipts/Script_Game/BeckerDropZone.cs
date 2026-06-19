using UnityEngine;
using UnityEngine.EventSystems;

public class BeckerDropZone : MonoBehaviour, IDropHandler
{
    public RecipeManager recipeManager;
    public Transform beckerPoint;

    public void OnDrop(PointerEventData eventData)
    {
        GameObject droppedItem = eventData.pointerDrag;

        if (droppedItem != null)
        {
            droppedItem.transform.position = beckerPoint.position;
            recipeManager.AddItem(droppedItem);
        }
    }
}