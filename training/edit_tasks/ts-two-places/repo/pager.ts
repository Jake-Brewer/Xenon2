/** Returns the items for one page. */
export function paginate<T>(items: T[], page: number): T[] {
  const start = page * 20;
  return items.slice(start, start + 20);
}

/** Returns the total number of pages. */
export function pageCount(total: number): number {
  return Math.ceil(total / 20);
}
