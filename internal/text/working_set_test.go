package text

import (
	"fmt"
	"strings"
	"testing"
)

// Fixed text and order reproduce the LRU cliff with a modest document, without
// depending on system fonts. BenchmarkLayoutWorkingSet also runs native shaping.
func layoutWorkingSet(rows int) []Params {
	ps := make([]Params, rows)
	for i := range ps {
		ps[i] = Params{Text: fmt.Sprintf("paragraph %03d: %s", i, strings.Repeat("word ", 60))}
	}
	return ps
}

func TestLayoutCacheRetainsLargeWorkingSet(t *testing.T) {
	s := newSystem()
	s.eng = &stubEngine{}
	ps := layoutWorkingSet(200)
	var first []*Layout
	var workingBytes int
	for _, p := range ps {
		first = append(first, s.Layout(p))
		workingBytes += s.layouts[p].bytes
	}
	if workingBytes <= maxLayoutBytes || workingBytes > maxLayoutBytes*3/2 {
		t.Fatalf("fixture needs %d bytes, want just above 8 MiB", workingBytes)
	}
	s.EndFrame()
	made := s.LayoutsMade()
	for frame := range 20 {
		for i, p := range ps {
			if s.Layout(p) != first[i] {
				t.Fatalf("frame %d reshaped row %d", frame, i)
			}
		}
		s.EndFrame()
	}
	if s.LayoutsMade() != made {
		t.Fatal("displayed working set was reshaped")
	}
}

func TestLayoutCacheReclaimsExcessAfterLaterFrames(t *testing.T) {
	s := newSystem()
	s.eng = &stubEngine{}
	for _, p := range layoutWorkingSet(200) {
		s.Layout(p)
	}
	for range liveLayoutFrames - 1 {
		s.EndFrame()
		if s.layoutBytes <= maxLayoutBytes {
			t.Fatal("reclaimed within live frame grace")
		}
	}
	// All windows share the monotonic clock: another window's next frame ages
	// this unused document too. No wall-clock timer runs for an idle window.
	s.Layout(Params{Text: "another window"})
	s.EndFrame()
	if s.frame != liveLayoutFrames || s.layoutBytes > maxLayoutBytes {
		t.Fatalf("frame %d retains %d bytes", s.frame, s.layoutBytes)
	}
	for range 241 {
		s.EndFrame()
	}
	if len(s.layouts) != 0 || s.layoutBytes != 0 {
		t.Fatal("old layouts survived expiry")
	}
}

func TestLayoutCacheCapsLiveBytes(t *testing.T) {
	s := newSystem()
	s.eng = &stubEngine{}
	first := s.Layout(Params{Text: "first"})
	for i := range 100 {
		s.Layout(Params{Text: fmt.Sprintf("%d: %s", i, strings.Repeat("word ", 2000))})
		if s.layoutBytes > maxLiveLayoutBytes {
			t.Fatalf("retained %d bytes", s.layoutBytes)
		}
	}
	if s.LayoutsMade() != 101 {
		t.Fatal("fixture should shape each unique paragraph")
	}
	if s.Layout(Params{Text: "first"}) == first {
		t.Fatal("hard cap did not evict oldest")
	}
	s.EndFrame()
	if s.layoutBytes > maxLiveLayoutBytes {
		t.Fatal("EndFrame exceeded hard cap")
	}
}

func TestLayoutCacheClearLargeWorkingSet(t *testing.T) {
	s := newSystem()
	s.eng = &stubEngine{}
	for _, p := range layoutWorkingSet(200) {
		s.Layout(p)
	}
	s.clearLayouts()
	if s.oldest != nil || s.newest != nil || len(s.layouts) != 0 || s.layoutBytes != 0 {
		t.Fatal("clear retained the large working set")
	}
}

func TestLayoutCacheLargeWorkingSetAcrossWindows(t *testing.T) {
	s := newSystem()
	s.eng = &stubEngine{}
	ps := layoutWorkingSet(200)
	first := s.Layout(ps[0])
	for _, p := range ps[1:] {
		s.Layout(p)
	}
	s.EndFrame()
	for range liveLayoutFrames - 2 {
		s.Layout(Params{Text: "another window"})
		s.EndFrame()
	}
	if s.Layout(ps[0]) != first {
		t.Fatal("alternating windows lost live text")
	}
	changed := ps[0]
	changed.Style.Size = 28
	if s.Layout(changed) == first {
		t.Fatal("font-size change reused stale geometry")
	}
	changed = ps[0]
	changed.Width = 100
	if s.Layout(changed) == first {
		t.Fatal("width change reused stale geometry")
	}
}

type workingSetForgetter struct {
	stubEngine
	forgotten bool
}

func (e *workingSetForgetter) fontCount() int { return maxFonts + 1 }
func (e *workingSetForgetter) forgetFonts()   { e.forgotten = true }

func TestLayoutCacheLargeWorkingSetFontGeneration(t *testing.T) {
	s := newSystem()
	e := &workingSetForgetter{}
	s.eng = e
	for _, p := range layoutWorkingSet(200) {
		s.Layout(p)
	}
	gen := s.Generation()
	s.EndFrame()
	if !e.forgotten || s.Generation() != gen+1 || len(s.layouts) != 0 || s.layoutBytes != 0 || s.oldest != nil || s.newest != nil {
		t.Fatal("font reset kept old generation layouts")
	}
}

// Each operation redraws one warmed viewport in the same order. Setup and
// initial shaping are excluded. Native uses the platform engine; Stub isolates
// cache policy. retained-B is the cache's conservative estimate, not heap RSS.
func BenchmarkLayoutWorkingSet(b *testing.B) {
	for _, native := range []bool{false, true} {
		engineName := "Stub"
		if native {
			engineName = "Native"
		}
		b.Run(engineName, func(b *testing.B) {
			for _, rows := range []int{40, 200} {
				b.Run(fmt.Sprintf("Rows%d", rows), func(b *testing.B) {
					s := newSystem()
					if !native {
						s.eng = &stubEngine{}
					}
					ps := layoutWorkingSet(rows)
					for _, p := range ps {
						s.Layout(p)
					}
					s.EndFrame()
					made := s.LayoutsMade()
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						for _, p := range ps {
							s.Layout(p)
						}
						s.EndFrame()
					}
					b.StopTimer()
					b.ReportMetric(float64(s.layoutBytes), "retained-B")
					b.ReportMetric(float64(s.LayoutsMade()-made)/float64(b.N), "shapes/op")
				})
			}
		})
	}
}
