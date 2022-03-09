package com.company.lib.util;

import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

public record Reversed<T>(List<T> original) implements Iterable<T> {

    public Iterator<T> iterator() {
        final ListIterator<T> i = original.listIterator(original.size());
        return new Iterator<T>() {
            public boolean hasNext() {
                return i.hasPrevious();
            }

            public T next() {
                return i.previous();
            }

            public void remove() {
                i.remove();
            }
        };
    }

    public static <T> Reversed<T> reversed(List<T> original) {
        return new Reversed<T>(original);
    }
}